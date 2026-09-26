"""Parity for the adaptive-mode rate arithmetic.

Every case is driven through the real `aiobotocore` / `botocore` classes with
a scripted clock injected, so the comparison is exact rather than a race
against `time.time()`.
"""

import numpy as np
import pytest

import mojo_aiobotocore as mab
from conftest import real_backoff, real_cubic, real_rate_clock, real_retry_quota, real_token_bucket

TICK, SET_RATE, ACQUIRE = 0, 1, 2


# ---------------------------------------------------------------------------
# AsyncTokenBucket
# ---------------------------------------------------------------------------

# The first op of every trace is the constructor's own set_max_rate, which is
# how the real AsyncTokenBucket seeds _last_timestamp and its fill rate.
BUCKET_TRACES = [
    [(SET_RATE, 1.0, 0.0)],
    [(SET_RATE, 1.0, 0.0), (TICK, 0.0, 1.0), (TICK, 0.0, 2.0)],
    [(SET_RATE, 1.0, 0.0), (ACQUIRE, 1.0, 0.0), (ACQUIRE, 1.0, 0.0)],
    [(SET_RATE, 1.0, 0.0), (ACQUIRE, 2.0, 0.0), (ACQUIRE, 1.0, 0.5), (ACQUIRE, 1.0, 1.0)],
    [(SET_RATE, 1.0, 0.0), (SET_RATE, 20.0, 0.0), (ACQUIRE, 5.0, 0.25), (ACQUIRE, 5.0, 0.5)],
    [(SET_RATE, 1.0, 0.0), (SET_RATE, 0.1, 0.0), (TICK, 0.0, 3.0), (ACQUIRE, 1.0, 3.0)],
    [(SET_RATE, 1.0, 0.0), (SET_RATE, 100.0, 0.0), (TICK, 0.0, 60.0), (ACQUIRE, 100.0, 60.0)],
    [(SET_RATE, 1.0, 0.0), (SET_RATE, 5.0, 0.0), (SET_RATE, 1.0, 1.0), (ACQUIRE, 3.0, 1.0)],
    [(SET_RATE, 1.0, 0.0), (SET_RATE, 2.0, 0.0), (TICK, 0.0, 0.5), (SET_RATE, 10.0, 0.5),
     (TICK, 0.0, 0.75), (ACQUIRE, 4.0, 0.75)],
    [(SET_RATE, 1.0, 0.0), (TICK, 0.0, 0.01), (ACQUIRE, 1.0, 0.01), (TICK, 0.0, 1.5),
     (ACQUIRE, 3.0, 1.5), (SET_RATE, 0.5, 2.0), (ACQUIRE, 1.0, 4.0)],
]


@pytest.mark.parametrize("trace", BUCKET_TRACES)
def test_token_bucket_matches_aiobotocore(trace):
    got = mab.token_bucket_trace(trace)
    expect = real_token_bucket(trace)
    np.testing.assert_allclose(got[:, 0], [c for c, _ in expect], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[:, 1], [s for _, s in expect], rtol=1e-12, atol=1e-12)


def test_token_bucket_sleep_amount_is_the_documented_formula():
    """(amount - capacity) / fill_rate, straight from `_sleep_amount`."""
    trace = [(SET_RATE, 1.0, 0.0), (SET_RATE, 4.0, 0.0), (TICK, 0.0, 0.1), (ACQUIRE, 3.0, 0.1)]
    got = mab.token_bucket_trace(trace)
    # capacity at 0.1s is 4 * 0.1 = 0.4, so 3 tokens need (3 - 0.4) / 4 = 0.65s
    assert got[2, 0] == pytest.approx(0.4)
    assert got[3, 1] == pytest.approx((3.0 - 0.4) / 4.0)


def test_token_bucket_min_rate_floor():
    """A rate below `_MIN_RATE` is raised to it. The capacity is also clamped
    to 1 because `value < 1` forces `_max_capacity = 1`, so the sleep amount is
    what shows the floored fill rate: (1 - 1/0.5) / 0.5 with capacity 0.5."""
    trace = [(SET_RATE, 1.0, 0.0), (SET_RATE, 0.1, 0.0), (TICK, 0.0, 1.0), (ACQUIRE, 1.0, 1.0)]
    got = mab.token_bucket_trace(trace)
    # after 1s at the floored rate the bucket holds 0.5 tokens, capped by the
    # max_capacity of 1 that a sub-1 rate forces
    assert got[2, 0] == pytest.approx(0.5)
    assert got[3, 1] == pytest.approx((1.0 - 0.5) / 0.5)

    trace2 = [(SET_RATE, 1.0, 0.0), (SET_RATE, 0.1, 0.0), (ACQUIRE, 1.0, 0.0)]
    got2 = mab.token_bucket_trace(trace2)
    assert got2[2, 1] == pytest.approx(1.0 / 0.5)  # (1 - 0) / _MIN_RATE


def test_token_bucket_capacity_is_capped():
    trace = [(SET_RATE, 1.0, 0.0), (SET_RATE, 2.0, 0.0), (TICK, 0.0, 100.0)]
    got = mab.token_bucket_trace(trace)
    assert got[2, 0] == pytest.approx(2.0)


def test_token_bucket_scaling_down_trims_capacity():
    trace = [
        (SET_RATE, 1.0, 0.0),
        (SET_RATE, 50.0, 0.0),
        (TICK, 0.0, 10.0),
        (SET_RATE, 1.0, 10.0),
    ]
    got = mab.token_bucket_trace(trace)
    assert got[2, 0] == pytest.approx(50.0)
    assert got[3, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CUBIC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("w_max", [0.0, 1.0, 5.5, 20.0, 137.25])
def test_cubic_zero_point_matches_botocore(w_max):
    theirs = real_cubic(w_max)
    # the cube root is a transcendental: Mojo's pow and CPython's ** differ in
    # the last few ulps, so this is a tolerance rather than an equality
    assert mab.cubic_zero_point(w_max) == pytest.approx(theirs._k, rel=1e-9)


@pytest.mark.parametrize(
    "dt", [0.0, 0.1, 0.5, 1.0, 2.5, 10.0, 100.0]
)
def test_cubic_success_matches_botocore(dt):
    start = 3.0
    theirs = real_cubic(20.0, start_time=start)
    got = mab.cubic_success(20.0, theirs._k, start, start + dt)
    assert got == pytest.approx(theirs.success_received(start + dt), rel=1e-9)


def test_cubic_error_matches_botocore():
    theirs = real_cubic(20.0, start_time=0.0)
    rate, (w_max, k, last_fail) = mab.cubic_error(7.5, 4.25)
    expect_rate = theirs.error_received(7.5, 4.25)
    expect = theirs.get_params_snapshot()
    assert rate == pytest.approx(expect_rate, rel=1e-12)
    assert w_max == pytest.approx(expect.w_max, rel=1e-12)
    assert k == pytest.approx(expect.k, rel=1e-9)
    assert last_fail == pytest.approx(expect.last_fail, rel=1e-12)


def test_cubic_error_updates_the_zero_point():
    """After an error the new k is derived from the new w_max, not left over."""
    before = mab.cubic_zero_point(20.0)
    _rate, (_w, k, _lf) = mab.cubic_error(7.5, 4.25)
    assert k == pytest.approx(mab.cubic_zero_point(7.5))
    assert k != pytest.approx(before)


def test_cubic_success_is_monotonic_after_the_zero_point():
    rates = [mab.cubic_success(10.0, mab.cubic_zero_point(10.0), 0.0, dt) for dt in (1, 2, 3, 4)]
    assert rates == sorted(rates)


# ---------------------------------------------------------------------------
# Truncated exponential backoff
# ---------------------------------------------------------------------------


def test_backoff_matches_botocore_with_a_fixed_jitter():
    jitter = [0.0, 0.25, 0.5, 0.75, 0.99, 0.1]
    attempts = [1, 2, 3, 4, 5, 6]
    got = mab.backoff_delays(attempts, jitter=jitter)
    expect = real_backoff(attempts, jitter)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_backoff_growth_is_exponential_then_capped():
    jitter = [1.0] * 12
    attempts = list(range(1, 13))
    got = mab.backoff_delays(attempts, jitter=jitter)
    # 1, 2, 4, 8, 16, 20 (capped), 20 ...
    assert got.tolist() == [1.0, 2.0, 4.0, 8.0, 16.0] + [20.0] * 7


def test_backoff_respects_a_custom_cap():
    got = mab.backoff_delays([1, 2, 3, 4], base=3, max_backoff=5.0, jitter=[1.0] * 4)
    assert got.tolist() == [1.0, 3.0, 5.0, 5.0]


def test_backoff_first_delay_is_pure_jitter():
    got = mab.backoff_delays([1], jitter=[0.375])
    assert got.tolist() == [0.375]


# ---------------------------------------------------------------------------
# Bucketed request-rate estimator
# ---------------------------------------------------------------------------

CLOCK_TRACES = [
    [0.0, 0.1, 0.2, 0.6, 1.0, 1.4, 2.0],
    [0.0, 0.51, 0.52, 1.01, 1.51, 3.0],
    [5.0, 5.0, 5.0, 5.5, 5.5, 6.0, 6.5, 7.0],
    [0.0, 0.49, 0.5, 0.51, 0.99, 1.0, 1.01],
    [10.0 + i * 0.17 for i in range(30)],
]


@pytest.mark.parametrize("times", CLOCK_TRACES)
def test_rate_clock_matches_botocore(times):
    amounts = [1] * len(times)
    got = mab.rate_clock(times, amounts)
    expect = real_rate_clock(times, amounts)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_rate_clock_stays_zero_inside_one_bucket():
    times = [0.0, 0.1, 0.2, 0.49]
    got = mab.rate_clock(times, [1] * 4)
    assert got.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_rate_clock_measures_the_bucket_rate():
    # five records land in the first 0.5s bucket, then one crosses the boundary
    times = [0.0, 0.1, 0.2, 0.3, 0.6]
    got = mab.rate_clock(times, [1] * 5)
    # 5 requests / 0.5s = 10/s, smoothed with 0.8 against a zero previous rate
    assert got[-1] == pytest.approx(10.0 * 0.8)


def test_rate_clock_honours_a_custom_smoothing():
    times = [0.0, 0.1, 0.6]
    got = mab.rate_clock(times, [1, 1, 1], smoothing=0.5)
    assert got[-1] == pytest.approx(6.0 * 0.5)


def test_rate_clock_honours_a_custom_bucket_range():
    times = [0.0, 0.1, 1.2, 2.4]
    got = mab.rate_clock(times, [1, 1, 1, 1], bucket_range=1.0)
    assert got[-1] > 0.0


# ---------------------------------------------------------------------------
# Retry quota
# ---------------------------------------------------------------------------

QUOTA_TRACES = [
    [(0, 5)],
    [(0, 5), (0, 5), (0, 5)],
    [(0, 400), (0, 400), (0, 400)],
    [(0, 5), (1, 5), (0, 5)],
    [(0, 5), (1, 5), (1, 5), (1, 5)],
    [(0, 10), (1, 3), (0, 7), (1, 100), (0, 1)],
    [(1, 10)],
]


@pytest.mark.parametrize("trace", QUOTA_TRACES)
def test_retry_quota_matches_botocore(trace):
    got, refused = mab.retry_quota_trace(trace)
    expect = real_retry_quota(trace)
    assert got[:, 0].tolist() == [a for a, _ in expect]
    assert got[:, 1].tolist() == [g for _, g in expect]
    assert refused == any(g == 0 for _, g in expect)


def test_retry_quota_refuses_past_capacity():
    _got, refused = mab.retry_quota_trace([(0, 10), (0, 10)], initial_capacity=15)
    assert refused is True


def test_retry_quota_release_never_exceeds_max():
    _got, refused = mab.retry_quota_trace([(0, 3), (1, 100)], initial_capacity=10)
    assert refused is False
    got, _ = mab.retry_quota_trace([(0, 3), (1, 100)], initial_capacity=10)
    assert got[1, 0] == 10
