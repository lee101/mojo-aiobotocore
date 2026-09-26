"""Compiled numeric kernels for the aiobotocore retry and checksum subset.

aiobotocore is an asyncio re-implementation of botocore. Nearly all of it is
plumbing: the endpoint provider, the event system, the credential chain, the
connection pool and the async/await plumbing around botocore. Three things in
it are genuinely numeric and are what this compilation unit implements:

  * the flexible-checksum accumulators (CRC-32 and CRC-32C) that
    `aiobotocore.httpchecksum` updates on every response byte, together with
    the aws-chunked framing arithmetic of `AioAwsChunkedWrapper._make_chunk`;
  * the client-side rate limiter arithmetic in
    `aiobotocore.retries.bucket.AsyncTokenBucket` -- the refill and sleep
    computations;
  * the CUBIC window-growth, the bucketed request-rate estimator and the
    truncated exponential backoff those two drive, plus the integer retry
    quota accounting.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.math import floor


def uptr(addr: Int) -> Pointer[UInt8, AnyOrigin[mut=True]]:
    return Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=addr)


def lptr(addr: Int) -> Pointer[Int64, AnyOrigin[mut=True]]:
    return Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=addr)


def u32ptr(addr: Int) -> Pointer[UInt32, AnyOrigin[mut=True]]:
    return Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=addr)


def f64ptr(addr: Int) -> Pointer[Float64, AnyOrigin[mut=True]]:
    return Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# CRC-32 (ISO-HDLC, the zlib/botocore polynomial) and CRC-32C (Castagnoli)
# ---------------------------------------------------------------------------

comptime CRC32_POLY = UInt32(0xEDB88320)
comptime CRC32C_POLY = UInt32(0x82F63B78)


@export("ab_crc_table")
def ab_crc_table(table_addr: Int, poly: Int) abi("C"):
    """Fill 256 UInt32 entries with the reflected CRC table for `poly`.

    botocore builds the same table from `crc32._crc32` / `crc32c`; the caller
    owns the buffer so no allocation crosses the ABI.
    """
    var t = u32ptr(table_addr)
    var p = UInt32(poly)
    for i in range(256):
        var crc = UInt32(i)
        for _ in range(8):
            if (crc & UInt32(1)) == UInt32(1):
                crc = (crc >> UInt32(1)) ^ p
            else:
                crc = crc >> UInt32(1)
        t[unsafe_offset=i] = crc


@export("ab_crc32")
def ab_crc32(buf: Int, n: Int, seed: Int, table_addr: Int) abi("C") -> UInt32:
    """Streaming CRC-32 over `n` bytes, continuing from `seed`.

    `seed` and the result are the "already processed" form, matching
    `zlib.crc32(chunk, previous)` and therefore `botocore.httpchecksum.
    Crc32Checksum.update`.
    """
    if n <= 0:
        return UInt32(seed)
    var p = uptr(buf)
    var t = u32ptr(table_addr)
    var crc = ~UInt32(seed)
    for i in range(n):
        var idx = Int((crc ^ UInt32(p[unsafe_offset=i])) & UInt32(0xFF))
        crc = t[unsafe_offset=idx] ^ (crc >> UInt32(8))
    return ~crc


@export("ab_crc32c")
def ab_crc32c(buf: Int, n: Int, seed: Int, table_addr: Int) abi("C") -> UInt32:
    """Streaming CRC-32C (Castagnoli) over `n` bytes, continuing from `seed`."""
    if n <= 0:
        return UInt32(seed)
    var p = uptr(buf)
    var t = u32ptr(table_addr)
    var crc = ~UInt32(seed)
    for i in range(n):
        var idx = Int((crc ^ UInt32(p[unsafe_offset=i])) & UInt32(0xFF))
        crc = t[unsafe_offset=idx] ^ (crc >> UInt32(8))
    return ~crc


# ---------------------------------------------------------------------------
# aws-chunked transfer framing
# ---------------------------------------------------------------------------


@export("ab_hex_len")
def ab_hex_len(n: Int, out_addr: Int) abi("C") -> Int:
    """Write `n` in lowercase hex with no leading zeros, as
    `aiobotocore.httpchecksum.AioAwsChunkedWrapper._make_chunk` does.

    Returns the number of bytes written; the buffer must hold 17 of them, which
    is the most an Int64 can take.
    """
    var o = uptr(out_addr)
    if n == 0:
        o[unsafe_offset=0] = 48
        return 1
    var digits = 0
    var v = n
    while v > 0:
        digits += 1
        v = v // 16
    var i = 0
    v = n
    while i < digits:
        var shift = (digits - 1 - i) * 4
        var d = (n >> shift) & 15
        var c = 48 + d
        if d >= 10:
            c = 87 + d
        o[unsafe_offset=i] = UInt8(c)
        i += 1
    return digits


@export("ab_aws_chunked_encode")
def ab_aws_chunked_encode(
    src: Int,
    n: Int,
    chunk_size: Int,
    dst: Int,
    out_cap: Int,
    crc_table_addr: Int,
    crc_out: Int,
) abi("C") -> Int:
    """Encode `n` bytes as an aws-chunked body, with the CRC-32 alongside.

    Each chunk is `hex(len)\\r\\n<data>\\r\\n`, and the body ends with one
    final empty chunk; this is
    the framing `AioAwsChunkedWrapper._make_chunk` writes. The CRC-32 of the
    whole plaintext body is written to `crc_out`, which is the accumulator
    `StreamingChecksumBody` validates. Returns the number of bytes written, or
    -1 if `out_cap` is too small.
    """
    var p = uptr(src)
    var o = uptr(dst)
    var crc_addr = u32ptr(crc_out)
    var written = 0
    var crc = ~UInt32(0)
    var pos = 0
    while pos < n:
        var take = chunk_size
        if pos + take > n:
            take = n - pos
        var digits = ab_hex_len(take, dst + written)
        o[unsafe_offset=written + digits] = 13
        o[unsafe_offset=written + digits + 1] = 10
        written += digits + 2
        if written + take + 2 > out_cap:
            return -1
        for k in range(take):
            var byte = p[unsafe_offset=pos + k]
            o[unsafe_offset=written + k] = byte
            var t = u32ptr(crc_table_addr)
            var idx = Int((crc ^ UInt32(byte)) & UInt32(0xFF))
            crc = t[unsafe_offset=idx] ^ (crc >> UInt32(8))
        written += take
        o[unsafe_offset=written] = 13
        o[unsafe_offset=written + 1] = 10
        written += 2
        pos += take
    # the terminator is one final empty chunk: "0\r\n" plus the CRLF that
    # closes it, which is what b"%s\r\n%s\r\n" % (b"0", b"") produces
    if written + 5 > out_cap:
        return -1
    o[unsafe_offset=written] = 48
    o[unsafe_offset=written + 1] = 13
    o[unsafe_offset=written + 2] = 10
    o[unsafe_offset=written + 3] = 13
    o[unsafe_offset=written + 4] = 10
    written += 5
    crc_addr[unsafe_offset=0] = ~crc
    return written


# ---------------------------------------------------------------------------
# AsyncTokenBucket refill / sleep arithmetic
# ---------------------------------------------------------------------------

# op codes: 0 tick (arg unused), 1 set_max_rate(arg), 2 acquire(arg)
# ops layout: 3 Float64 per op -- (kind, arg, timestamp)
# out layout: 2 Float64 per op -- (capacity after, sleep amount or -1)


@export("ab_token_bucket_run")
def ab_token_bucket_run(
    ops_addr: Int,
    n: Int,
    min_rate: Float64,
    init_cap: Float64,
    out_addr: Int,
) abi("C") -> Int:
    """Replay `AsyncTokenBucket` over a scripted operation/timestamp trace.

    `min_rate` is the bucket's `_MIN_RATE` floor (0.5 upstream). The bucket
    starts empty, as `AsyncTokenBucket.__init__` leaves it, so the trace's
    first op is the constructor's own `set_max_rate`. Returns 0.
    """
    var ops = f64ptr(ops_addr)
    var o = f64ptr(out_addr)
    var fill_rate = min_rate
    var max_capacity = 1.0
    var capacity = 0.0
    var last_ts = 0.0
    var have_last = 0
    for i in range(n):
        var kind = Int(ops[unsafe_offset=3 * i + 0])
        var arg = ops[unsafe_offset=3 * i + 1]
        var ts = ops[unsafe_offset=3 * i + 2]
        # _refill
        if have_last == 1:
            var filled = capacity + (ts - last_ts) * fill_rate
            if filled > max_capacity:
                filled = max_capacity
            capacity = filled
        last_ts = ts
        have_last = 1
        if kind == 1:
            # _set_max_rate: max(value, _min_rate), not max(value, current)
            if arg > min_rate:
                fill_rate = arg
            else:
                fill_rate = min_rate
            if arg >= 1.0:
                max_capacity = arg
            else:
                max_capacity = 1.0
            if capacity > max_capacity:
                capacity = max_capacity
            o[unsafe_offset=2 * i + 0] = capacity
            o[unsafe_offset=2 * i + 1] = -1.0
        elif kind == 2:
            var sleep_amount = -1.0
            if arg <= capacity:
                capacity = capacity - arg
            else:
                sleep_amount = (arg - capacity) / fill_rate
            o[unsafe_offset=2 * i + 0] = capacity
            o[unsafe_offset=2 * i + 1] = sleep_amount
        else:
            o[unsafe_offset=2 * i + 0] = capacity
            o[unsafe_offset=2 * i + 1] = -1.0
    return 0


# ---------------------------------------------------------------------------
# CUBIC window growth (botocore.retries.throttling.CubicCalculator)
# ---------------------------------------------------------------------------


@export("ab_cubic_zero_point")
def ab_cubic_zero_point(
    w_max: Float64, scale_constant: Float64, beta: Float64
) abi("C") -> Float64:
    """`((w_max * (1 - beta)) / scale_constant) ** (1/3)`, the CUBIC zero point."""
    var scaled = (w_max * (1.0 - beta)) / scale_constant
    if scaled <= 0.0:
        return 0.0
    return pow(scaled, 1.0 / 3.0)


@export("ab_cubic_success")
def ab_cubic_success(
    w_max: Float64, k: Float64, last_fail: Float64, timestamp: Float64,
    scale_constant: Float64
) abi("C") -> Float64:
    """`scale_constant * (dt - k) ** 3 + w_max`, the rate after a success."""
    var dt = timestamp - last_fail
    var x = dt - k
    return scale_constant * (x * x * x) + w_max


@export("ab_cubic_error")
def ab_cubic_error(
    w_max: Float64,
    current_rate: Float64,
    timestamp: Float64,
    scale_constant: Float64,
    beta: Float64,
    out_addr: Int,
) abi("C") -> Float64:
    """Fold an error response into the CUBIC state.

    Writes the updated (w_max, k, last_fail) to `out` as three Float64 and
    returns `current_rate * beta`, which is the rate to retry at.
    """
    var o = f64ptr(out_addr)
    var scaled = (current_rate * (1.0 - beta)) / scale_constant
    var k = 0.0
    if scaled > 0.0:
        k = pow(scaled, 1.0 / 3.0)
    o[unsafe_offset=0] = current_rate
    o[unsafe_offset=1] = k
    o[unsafe_offset=2] = timestamp
    return current_rate * beta


# ---------------------------------------------------------------------------
# Truncated exponential backoff (botocore.retries.standard.ExponentialBackoff)
# ---------------------------------------------------------------------------


@export("ab_backoff_delays")
def ab_backoff_delays(
    attempts_addr: Int,
    n: Int,
    base: Int,
    max_backoff: Float64,
    jitter_addr: Int,
    out_addr: Int,
) abi("C") -> Int:
    """`jitter[i] * min(base ** (attempt[i] - 1), max_backoff)` for each entry.

    `attempts` holds 1-based attempt numbers, exactly as
    `RetryContext.attempt_number` does. Returns 0.
    """
    var a = lptr(attempts_addr)
    var j = f64ptr(jitter_addr)
    var o = f64ptr(out_addr)
    for i in range(n):
        var attempt = a[unsafe_offset=i]
        var e = Int(attempt) - 1
        var grow = 1.0
        var k = 0
        while k < e:
            grow = grow * Float64(base)
            k += 1
        var capped = grow
        if capped > max_backoff:
            capped = max_backoff
        o[unsafe_offset=i] = j[unsafe_offset=i] * capped
    return 0


# ---------------------------------------------------------------------------
# Bucketed request-rate estimator (botocore.retries.adaptive.RateClocker)
# ---------------------------------------------------------------------------


@export("ab_rate_clock")
def ab_rate_clock(
    times_addr: Int,
    amounts_addr: Int,
    n: Int,
    smoothing: Float64,
    bucket_range: Float64,
    out_addr: Int,
) abi("C") -> Int:
    """Replay `RateClocker.record(amount)` over a scripted clock.

    Writes the measured rate after each record to `out`. The first call seeds
    `last_bucket` from the clock, so `times[0]` behaves as the constructor's
    initial read.
    """
    var t = f64ptr(times_addr)
    var a = lptr(amounts_addr)
    var o = f64ptr(out_addr)
    var scale = 1.0 / bucket_range
    var measured = 0.0
    var last_bucket = floor(t[unsafe_offset=0] * scale) / scale
    var count = 0.0
    for i in range(n):
        var now = t[unsafe_offset=i]
        var bucket = floor(now * scale) / scale
        count += Float64(a[unsafe_offset=i])
        if bucket > last_bucket:
            var current_rate = count / (bucket - last_bucket)
            measured = current_rate * smoothing + measured * (1.0 - smoothing)
            count = 0.0
            last_bucket = bucket
        o[unsafe_offset=i] = measured
    return 0


# ---------------------------------------------------------------------------
# Retry quota accounting (botocore.retries.quota.RetryQuota)
# ---------------------------------------------------------------------------

# op codes: 0 acquire, 1 release. ops layout: 2 Int64 per op -- (kind, amount)
# out layout: 2 Int64 per op -- (available after, granted 0/1)


@export("ab_retry_quota_run")
def ab_retry_quota_run(
    ops_addr: Int, n: Int, initial: Int, out_addr: Int
) abi("C") -> Int:
    """Replay `RetryQuota.acquire` / `RetryQuota.release` over a trace.

    Returns 0, or -1 if an acquire was refused for lack of capacity.
    """
    var ops = lptr(ops_addr)
    var o = lptr(out_addr)
    var max_capacity = initial
    var available = initial
    var failed = 0
    for i in range(n):
        var kind = Int(ops[unsafe_offset=2 * i + 0])
        var amount = Int(ops[unsafe_offset=2 * i + 1])
        if kind == 0:
            if amount > available:
                o[unsafe_offset=2 * i + 0] = Int64(available)
                o[unsafe_offset=2 * i + 1] = 0
                failed = 1
            else:
                available = available - amount
                o[unsafe_offset=2 * i + 0] = Int64(available)
                o[unsafe_offset=2 * i + 1] = 1
        else:
            # release() returns None upstream: it either gave capacity back or
            # was a no-op because the bucket was already full. Both count as
            # a satisfied operation, so `granted` stays 1.
            if max_capacity != available:
                var room = max_capacity - available
                var give = amount
                if give > room:
                    give = room
                available = available + give
            o[unsafe_offset=2 * i + 0] = Int64(available)
            o[unsafe_offset=2 * i + 1] = 1
    return -1 if failed == 1 else 0
