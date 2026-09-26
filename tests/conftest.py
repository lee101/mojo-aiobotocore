import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-aiobotocore.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-aiobotocore.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Drivers for the real aiobotocore / botocore arithmetic.
#
# The parity tests compare against these, not against a transcription written
# in the test file. Where a class reads the wall clock, a scripted clock is
# injected so the comparison is exact rather than approximate.
# ---------------------------------------------------------------------------


class ScriptedClock:
    """A `botocore.retries.bucket.Clock` whose time is set explicitly.

    The caller parks `now` before every call, so no read order is implicit and
    the comparison with the kernel is exact.
    """

    def __init__(self, now=0.0):
        self.now = now

    def current_time(self):
        return self.now

    def sleep(self, amount):
        raise AssertionError("the parity harness must never sleep")


def real_token_bucket(ops, max_rate=1.0, min_rate=0.5):
    """Replay `aiobotocore.retries.bucket.AsyncTokenBucket` over a trace.

    `ops` is a list of ``(kind, arg, timestamp)`` with kind 0 tick, 1
    set_max_rate, 2 acquire. The first op is applied by the constructor, which
    is how the real object seeds `_last_timestamp` and its fill rate. Returns
    ``(capacity, sleep_or_minus_one)`` per op.
    """
    from aiobotocore.retries.bucket import AsyncTokenBucket

    clock = ScriptedClock(ops[0][2])
    bucket = AsyncTokenBucket(max_rate=max_rate, clock=clock, min_rate=min_rate)
    trace = [(bucket.available_capacity, -1.0)]
    for kind, arg, ts in ops[1:]:
        clock.now = ts
        sleep = -1.0
        if kind == 1:
            bucket._set_max_rate(arg)
        elif kind == 2:
            bucket._refill()
            if arg <= bucket.available_capacity:
                bucket._current_capacity -= arg
            else:
                sleep = bucket._sleep_amount(arg)
        else:
            bucket._refill()
        trace.append((bucket.available_capacity, sleep))
    return trace


def real_cubic(w_max, scale_constant=0.4, beta=0.7, start_time=0.0):
    """A real `botocore.retries.throttling.CubicCalculator`."""
    from botocore.retries.throttling import CubicCalculator

    return CubicCalculator(
        starting_max_rate=w_max,
        start_time=start_time,
        scale_constant=scale_constant,
        beta=beta,
    )


def real_backoff(attempts, jitter, max_backoff=20.0):
    """Truncated exponential backoff from botocore's own class."""
    from botocore.retries.standard import ExponentialBackoff

    class _Ctx:
        def __init__(self, attempt_number):
            self.attempt_number = attempt_number

    it = iter(jitter)
    backoff = ExponentialBackoff(max_backoff=max_backoff, random=lambda: next(it))
    return [backoff.delay_amount(_Ctx(a)) for a in attempts]


def real_rate_clock(times, amounts, smoothing=0.8, bucket_range=0.5):
    """A real `botocore.retries.adaptive.RateClocker` on a scripted clock."""
    from botocore.retries.adaptive import RateClocker

    clock = ScriptedClock(times[0])
    # the constructor reads the clock once to seed the first bucket
    clocker = RateClocker(clock, smoothing=smoothing, time_bucket_range=bucket_range)
    out = []
    for t, amount in zip(times, amounts):
        clock.now = t
        out.append(clocker.record(amount))
    return out


def real_retry_quota(ops, initial_capacity=500):
    """A real `botocore.retries.quota.RetryQuota`."""
    from botocore.retries.quota import RetryQuota

    quota = RetryQuota(initial_capacity=initial_capacity)
    trace = []
    for kind, amount in ops:
        if kind == 0:
            granted = quota.acquire(amount)
        else:
            quota.release(amount)
            granted = True
        trace.append((quota.available_capacity, 1 if granted else 0))
    return trace
