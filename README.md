# mojo-aiobotocore

`mojo-aiobotocore` is a Mojo port of the numeric core of
[aiobotocore](https://aiobotocore.readthedocs.io/) 2.25 — the flexible-checksum
accumulators and aws-chunked framing of its response layer, and the retry and
adaptive-mode rate arithmetic underneath it.

**aiobotocore is overwhelmingly I/O plumbing.** It is an asyncio
re-implementation of botocore: the endpoint provider and rule engine, the event
system, the credential providers, the JSON/XML parsers, the connection pooling,
the waiter and pagination state machines and the async/await plumbing around all
of it. None of that is ported, and none of it is numeric. What *is* numeric
inside aiobotocore is a small, well-defined set of loops, and that is what this
project ports.

The Python package is `mojo_aiobotocore`, so it installs alongside the real
`aiobotocore` and the parity tests compare the two directly.

## Covered subset

| area | upstream source | ported API |
| --- | --- | --- |
| CRC-32 (ISO-HDLC) | `aiobotocore.httpchecksum` via `botocore.httpchecksum.Crc32Checksum` (`zlib.crc32`) | `crc32`, table builder |
| CRC-32C (Castagnoli) | the `crc32c` package botocore's CRT checksum class delegates to | `crc32c`, table builder |
| aws-chunked framing | `aiobotocore.httpchecksum.AioAwsChunkedWrapper._make_chunk` | `aws_chunked_encode`, `hex_len` |
| Client rate limiter | `aiobotocore.retries.bucket.AsyncTokenBucket` | `token_bucket_trace` |
| CUBIC window growth | `botocore.retries.throttling.CubicCalculator` | `cubic_zero_point`, `cubic_success`, `cubic_error` |
| Request-rate estimator | `botocore.retries.adaptive.RateClocker` | `rate_clock` |
| Exponential backoff | `botocore.retries.standard.ExponentialBackoff` | `backoff_delays` |
| Retry quota | `botocore.retries.quota.RetryQuota` | `retry_quota_trace` |

**Not implemented, and not attempted:** endpoint resolution and rules,
credentials and signing (SigV4/HMAC live in botocore and are string and hash
plumbing, not loops), request signing, XML/JSON parsing, event hooks,
connection pooling, `waiter`, `paginate`, retries *policy* (which is control
flow, not arithmetic), and anything `async`. Use the real `aiobotocore` for all
of it.

The checksums and the quota are exact integer algorithms and are asserted with
exact equality. The rate arithmetic is floating point and is asserted to
`rtol=1e-12`, except the CUBIC zero point, which involves a cube root: Mojo's
`pow` and CPython's `**` differ in the last few ulps there, so that one is
`rtol=1e-9` and says so.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aiobotocore.so
PYTHONPATH=python python -m pytest tests -q
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo = "==1.2.0.dev2026092605"`). Do not run `pixi install` in this tree; the
shared environment at `/nvme0n1-disk/mojo-toolchain` is the environment.

```python
import mojo_aiobotocore as mab

mab.crc32(b"123456789")            # 0xCBF43926, same as zlib.crc32
mab.crc32c(b"123456789")           # 0xE3069283, the Castagnoli value
body, crc = mab.aws_chunked_encode(payload, chunk_size=8192)
mab.backoff_delays([1, 2, 3, 4], jitter=[0.5] * 4)   # [0.5, 1.0, 2.0, 4.0]
```

## Performance

Best-of-seven wall clock, same process, every case checked against its reference
before timing.

| case | reference | mojo-aiobotocore | result |
| --- | ---: | ---: | ---: |
| `crc32` n=4194304 | 3.35 ms | 12.97 ms | **0.26x, a slowdown** |
| `crc32c` n=4194304 | 0.22 ms | 13.00 ms | **0.02x, a large slowdown** |
| `crc32` vs a bit-at-a-time Python loop, n=65536 | 204.97 ms | 0.23 ms | 891x faster |
| `aws_chunked_encode` n=1048576 | 1.25 ms | 3.92 ms | **0.32x, a slowdown** |
| `token_bucket` trace, 65536 steps | 29.66 ms | 0.33 ms | 89x faster |
| `backoff_delays` n=65536 | 79.73 ms | 0.99 ms | 80x faster |

The checksum rows are real losses and there is no way to dress them up. The
references are not strawmen: `zlib.crc32` is zlib's slice-by-8 assembly, and the
`crc32c` package computes four interleaved CRC streams and folds them, which is
how a carry-less-multiply instruction reaches 19 GB/s. A table-driven CRC in
scalar code consumes one byte per table load and cannot compete with that; the
Mojo kernel runs at roughly 320 MB/s. Beating it would need either a
carry-less-multiply intrinsic or a slice-by-16 kernel, neither of which the
exported ABI can express here. The port is still worth 891x over what a
pure-Python port of the same algorithm costs, which is the comparison that
matters for anyone who would otherwise write the loop in Python.

The rate-arithmetic rows are the opposite story: those are scalar float loops
over a few thousand elements with no vectorisable structure, and the compiled
version is 80-90x faster because it removes the interpreter from the inner
loop entirely.

Reproduce with:

```bash
python bench/bench.py
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-aiobotocore.so`.

The `python/mojo_aiobotocore` layer owns every array, including the two 256-entry
CRC tables, which it builds once through the kernel's own `ab_crc_table` at
import time and then hands to every checksum call as an address. Buffers cross
the C ABI as 64-bit addresses (`ctypes.c_int64`; `c_int` truncates and
segfaults) and are reconstructed in Mojo as pointers, which keeps the exported
symbols non-parametric.

The CRC kernels are *streaming*: `crc32(chunk, seed)` returns the "already
processed" form that `zlib.crc32(chunk, previous)` uses, so
`StreamingChecksumBody`'s repeated `checksum.update(chunk)` calls chain
correctly. A kernel that reset its seed per call would agree with a single-shot
comparison and disagree with the chunked one, which is why the tests drive
chunk-by-chunk updates through the real `Crc32Checksum`.

The token-bucket kernel takes a scripted `(kind, arg, timestamp)` trace rather
than a wall clock. That is deliberate: the real `AsyncTokenBucket` reads
`clock.current_time()`, and a trace makes the comparison with the real class
exact instead of a race. The trace's first operation is the constructor's own
`set_max_rate`, because that is where the real object seeds `_last_timestamp`
and its fill rate.

## Tests

101 parity tests, run against the real classes:

- CRC-32 against `zlib.crc32` and `botocore.httpchecksum.Crc32Checksum`,
  including chunk-by-chunk streaming updates and empty chunks at 15 payload
  sizes up to 65537 bytes;
- CRC-32C against the `crc32c` package, plus the standard check vector
  `crc32c("123456789") == 0xE3069283`;
- both 256-entry tables against an independently written table builder, and
  the relation between a table entry and the single-byte digest, so a table
  built with the wrong polynomial cannot pass;
- aws-chunked framing driven through a real `AioAwsChunkedWrapper` at five
  payload/chunk-size combinations, plus the known-vector hex lengths;
- the token-bucket trace against `aiobotocore.retries.bucket.AsyncTokenBucket`
  on an explicitly parked clock over nine traces, plus the `_MIN_RATE` floor,
  the capacity cap and the scale-down trim;
- CUBIC against `botocore.retries.throttling.CubicCalculator` including a state
  snapshot after an error;
- backoff against `botocore.retries.standard.ExponentialBackoff` with an
  injected RNG, plus the exponential-then-capped shape;
- `RateClocker` and `RetryQuota` against botocore, driven by a scripted clock.

## License

MIT
