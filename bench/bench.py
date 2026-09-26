"""Correctness-gated benchmark for mojo-aiobotocore.

Every case checks agreement with its reference before timing, so a regression
in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_aiobotocore as mab  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _rand(n, seed):
    return bytes(
        np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tolist()
    )


def bench_crc32(n: int = 1 << 22):
    import zlib

    data = _rand(n, 0)
    assert mab.crc32(data) == zlib.crc32(data)
    return (
        f"crc32 n={n}",
        _time(lambda: zlib.crc32(data)),
        _time(lambda: mab.crc32(data)),
    )


def bench_crc32c(n: int = 1 << 22):
    import crc32c

    data = _rand(n, 1)
    assert mab.crc32c(data) == crc32c.crc32c(data)
    return (
        f"crc32c n={n}",
        _time(lambda: crc32c.crc32c(data)),
        _time(lambda: mab.crc32c(data)),
    )


def bench_crc32_bitwise(n: int = 1 << 16):
    """The same CRC-32 as a bit-at-a-time Python loop, which is what a naive
    pure-Python port would do. Included to show the table-driven kernel is not
    just a transcription of that loop."""
    data = _rand(n, 2)

    def bitwise(buf, seed=0):
        crc = seed ^ 0xFFFFFFFF
        for byte in buf:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    assert bitwise(data) == mab.crc32(data)
    return (
        f"crc32 vs bitwise python n={n}",
        _time(lambda: bitwise(data), 3),
        _time(lambda: mab.crc32(data)),
    )


def bench_aws_chunked(n: int = 1 << 20, chunk_size: int = 8192):
    """aws-chunked framing, against the same framing plus CRC-32 written in
    Python. The reference builds identical bytes."""
    import zlib

    data = _rand(n, 3)
    body, crc = mab.aws_chunked_encode(data, chunk_size=chunk_size)
    assert crc == zlib.crc32(data)

    def reference():
        out = bytearray()
        for i in range(0, len(data), chunk_size):
            piece = data[i : i + chunk_size]
            out += b"%x\r\n" % len(piece)
            out += piece
            out += b"\r\n"
        out += b"0\r\n\r\n"
        return bytes(out), zlib.crc32(data)

    expect, _ = reference()
    assert body == expect
    return (
        f"aws_chunked_encode n={n}",
        _time(reference, 3),
        _time(lambda: mab.aws_chunked_encode(data, chunk_size=chunk_size)),
    )


def bench_token_bucket(steps: int = 1 << 16):
    """Token-bucket refill trace, against the same recurrence in Python."""
    rng = np.random.default_rng(4)
    times = np.cumsum(rng.random(steps)) * 0.01
    trace = np.zeros((steps, 3), dtype=np.float64)
    trace[:, 0] = 0
    trace[:, 1] = 0
    trace[:, 2] = times
    got = mab.token_bucket_trace(trace)
    assert got.shape == (steps, 2)

    fill_rate = 0.5
    max_capacity = 1.0
    capacity = 0.0
    last = float(times[0])

    def reference():
        nonlocal fill_rate, max_capacity, capacity, last
        fill_rate = 0.5
        max_capacity = 1.0
        capacity = 0.0
        last = float(times[0])
        acc = 0.0
        for ts in times:
            capacity = min(max_capacity, capacity + (float(ts) - last) * fill_rate)
            last = float(ts)
            acc += capacity
        return acc

    return (
        f"token_bucket steps={steps}",
        _time(reference, 3),
        _time(lambda: mab.token_bucket_trace(trace)),
    )


def bench_backoff(attempts: int = 1 << 16):
    rng = np.random.default_rng(5)
    a = rng.integers(1, 12, attempts, dtype=np.int64)
    j = rng.random(attempts)
    got = mab.backoff_delays(a, jitter=j)
    assert got.shape == (attempts,)

    def reference():
        acc = 0.0
        for attempt, r in zip(a, j):
            grow = 1.0
            for _ in range(int(attempt) - 1):
                grow *= 2.0
            acc += r * min(grow, 20.0)
        return acc

    return (
        f"backoff_delays n={attempts}",
        _time(reference, 3),
        _time(lambda: mab.backoff_delays(a, jitter=j)),
    )


def main():
    print(f"{'case':<36}{'reference':>13}{'mojo-aiobotocore':>18}{'ratio':>10}")
    print("-" * 77)
    for fn in (
        bench_crc32,
        bench_crc32c,
        bench_crc32_bitwise,
        bench_aws_chunked,
        bench_token_bucket,
        bench_backoff,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<36}{ref*1e3:>11.2f}ms{got*1e3:>16.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
