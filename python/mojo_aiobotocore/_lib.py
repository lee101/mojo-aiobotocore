"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below stay `c_int64` for addresses; `c_int` truncates
them and segfaults. The CRC tables and every scratch buffer are allocated here
and handed to the kernel as addresses.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aiobotocore.so"

_I = ctypes.c_int64
_D = ctypes.c_double
_U32 = ctypes.c_uint32

# Reflected CRC-32 (ISO-HDLC) and CRC-32C (Castagnoli) polynomials.
CRC32_POLY = 0xEDB88320
CRC32C_POLY = 0x82F63B78

# AsyncTokenBucket._MIN_RATE
BUCKET_MIN_RATE = 0.5


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.ab_crc_table.restype = None
    lib.ab_crc_table.argtypes = [_I, _I]

    lib.ab_crc32.restype = _U32
    lib.ab_crc32.argtypes = [_I, _I, _I, _I]

    lib.ab_crc32c.restype = _U32
    lib.ab_crc32c.argtypes = [_I, _I, _I, _I]

    lib.ab_hex_len.restype = _I
    lib.ab_hex_len.argtypes = [_I, _I]

    lib.ab_aws_chunked_encode.restype = _I
    lib.ab_aws_chunked_encode.argtypes = [_I, _I, _I, _I, _I, _I, _I]

    lib.ab_token_bucket_run.restype = _I
    lib.ab_token_bucket_run.argtypes = [_I, _I, _D, _D, _I]

    lib.ab_cubic_zero_point.restype = _D
    lib.ab_cubic_zero_point.argtypes = [_D, _D, _D]

    lib.ab_cubic_success.restype = _D
    lib.ab_cubic_success.argtypes = [_D, _D, _D, _D, _D]

    lib.ab_cubic_error.restype = _D
    lib.ab_cubic_error.argtypes = [_D, _D, _D, _D, _D, _I]

    lib.ab_backoff_delays.restype = _I
    lib.ab_backoff_delays.argtypes = [_I, _I, _I, _D, _I, _I]

    lib.ab_rate_clock.restype = _I
    lib.ab_rate_clock.argtypes = [_I, _I, _I, _D, _D, _I]

    lib.ab_retry_quota_run.restype = _I
    lib.ab_retry_quota_run.argtypes = [_I, _I, _I, _I]
    return lib


lib = _load()


def _u8(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray) and buf.dtype == np.uint8 and buf.flags.c_contiguous:
        return buf
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return np.frombuffer(buf, dtype=np.uint8)
    return np.ascontiguousarray(buf, dtype=np.uint8)


def _crc_table(poly: int) -> np.ndarray:
    table = np.zeros(256, dtype=np.uint32)
    lib.ab_crc_table(table.ctypes.data, poly)
    return table


_CRC32_TABLE = _crc_table(CRC32_POLY)
_CRC32C_TABLE = _crc_table(CRC32C_POLY)


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def crc32(data, seed: int = 0) -> int:
    """Streaming CRC-32, the form `zlib.crc32(chunk, previous)` produces."""
    b = _u8(data)
    if b.size == 0:
        return seed & 0xFFFFFFFF
    return int(lib.ab_crc32(b.ctypes.data, b.size, seed, _CRC32_TABLE.ctypes.data))


def crc32c(data, seed: int = 0) -> int:
    """Streaming CRC-32C (Castagnoli), the `crc32c` package's form."""
    b = _u8(data)
    if b.size == 0:
        return seed & 0xFFFFFFFF
    return int(lib.ab_crc32c(b.ctypes.data, b.size, seed, _CRC32C_TABLE.ctypes.data))


def hex_len(n: int) -> bytes:
    """`hex(n)[2:]` as bytes, the chunk-length field of aws-chunked framing."""
    buf = np.zeros(20, dtype=np.uint8)
    written = lib.ab_hex_len(n, buf.ctypes.data)
    return buf[:written].tobytes()


def aws_chunked_encode(data, chunk_size: int = 8192) -> tuple[bytes, int]:
    """Encode `data` as an aws-chunked body; returns ``(body, crc32)``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    b = _u8(data)
    cap = b.size + 2 * (b.size // chunk_size + 1) * 12 + 64
    out = np.zeros(cap, dtype=np.uint8)
    crc = np.zeros(1, dtype=np.uint32)
    written = lib.ab_aws_chunked_encode(
        b.ctypes.data,
        b.size,
        chunk_size,
        out.ctypes.data,
        out.size,
        _CRC32_TABLE.ctypes.data,
        crc.ctypes.data,
    )
    if written < 0:
        raise RuntimeError("aws-chunked output buffer too small")
    return out[:written].tobytes(), int(crc[0])


# ---------------------------------------------------------------------------
# Adaptive-mode rate arithmetic
# ---------------------------------------------------------------------------

BUCKET_TICK, BUCKET_SET_RATE, BUCKET_ACQUIRE = 0, 1, 2


def token_bucket_trace(ops, min_rate: float = BUCKET_MIN_RATE):
    """Replay `AsyncTokenBucket` over ``(kind, arg, timestamp)`` operations.

    Returns an ``(n, 2)`` array of (capacity after, sleep amount); the sleep is
    -1.0 for operations that do not need one. The first op is the
    constructor's own `set_max_rate`, because the real class leaves the bucket
    empty and seeds `_last_timestamp` there.
    """
    trace = np.ascontiguousarray(ops, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(trace.shape[0] * 2, dtype=np.float64)
    lib.ab_token_bucket_run(
        trace.ctypes.data,
        trace.shape[0],
        ctypes.c_double(min_rate),
        ctypes.c_double(0.0),
        out.ctypes.data,
    )
    return out.reshape(-1, 2)


def cubic_zero_point(w_max: float, scale_constant: float = 0.4, beta: float = 0.7) -> float:
    return float(
        lib.ab_cubic_zero_point(
            ctypes.c_double(w_max), ctypes.c_double(scale_constant), ctypes.c_double(beta)
        )
    )


def cubic_success(
    w_max: float, k: float, last_fail: float, timestamp: float, scale_constant: float = 0.4
) -> float:
    return float(
        lib.ab_cubic_success(
            ctypes.c_double(w_max),
            ctypes.c_double(k),
            ctypes.c_double(last_fail),
            ctypes.c_double(timestamp),
            ctypes.c_double(scale_constant),
        )
    )


def cubic_error(
    current_rate: float,
    timestamp: float,
    scale_constant: float = 0.4,
    beta: float = 0.7,
) -> tuple[float, tuple[float, float, float]]:
    """Returns ``(retry_rate, (w_max, k, last_fail))`` after an error."""
    out = np.zeros(3, dtype=np.float64)
    rate = float(
        lib.ab_cubic_error(
            ctypes.c_double(0.0),  # the previous w_max is replaced outright
            ctypes.c_double(current_rate),
            ctypes.c_double(timestamp),
            ctypes.c_double(scale_constant),
            ctypes.c_double(beta),
            out.ctypes.data,
        )
    )
    return rate, (float(out[0]), float(out[1]), float(out[2]))


def backoff_delays(attempts, base: int = 2, max_backoff: float = 20.0, jitter=None):
    """Truncated exponential backoff delays, `jitter` drawn in [0, 1)."""
    a = np.ascontiguousarray(attempts, dtype=np.int64).reshape(-1)
    j = (
        np.ascontiguousarray(jitter, dtype=np.float64).reshape(-1)
        if jitter is not None
        else np.zeros(a.size, dtype=np.float64)
    )
    out = np.zeros(a.size, dtype=np.float64)
    lib.ab_backoff_delays(
        a.ctypes.data, a.size, base, ctypes.c_double(max_backoff), j.ctypes.data, out.ctypes.data
    )
    return out


def rate_clock(times, amounts=1, smoothing: float = 0.8, bucket_range: float = 0.5):
    """Replay `RateClocker.record` over a scripted clock."""
    t = np.ascontiguousarray(times, dtype=np.float64).reshape(-1)
    a = np.ascontiguousarray(amounts, dtype=np.int64).reshape(-1)
    if a.size == 1 and t.size > 1:
        a = np.full(t.size, int(a[0]), dtype=np.int64)
    out = np.zeros(t.size, dtype=np.float64)
    lib.ab_rate_clock(
        t.ctypes.data,
        a.ctypes.data,
        t.size,
        ctypes.c_double(smoothing),
        ctypes.c_double(bucket_range),
        out.ctypes.data,
    )
    return out


QUOTA_ACQUIRE, QUOTA_RELEASE = 0, 1


def retry_quota_trace(ops, initial_capacity: int = 500) -> tuple[np.ndarray, bool]:
    """Replay `RetryQuota` acquire/release; returns the trace and whether any
    acquire was refused."""
    trace = np.ascontiguousarray(ops, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(trace.shape[0] * 2, dtype=np.int64)
    rc = lib.ab_retry_quota_run(trace.ctypes.data, trace.shape[0], int(initial_capacity), out.ctypes.data)
    return out.reshape(-1, 2), rc != 0
