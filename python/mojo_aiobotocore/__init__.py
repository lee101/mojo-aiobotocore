"""mojo-aiobotocore: the numeric core of aiobotocore's retry and checksum layer.

aiobotocore is an asyncio re-implementation of botocore. The endpoint
provider, the event system, the credential providers, the connection pooling
and the async plumbing are not ported and are not part of this package's API.
What is ported is the numeric work that layer actually performs: the flexible
checksum accumulators and aws-chunked framing of `aiobotocore.httpchecksum`,
the token-bucket refill arithmetic of `aiobotocore.retries.bucket`, and the
CUBIC growth, request-rate estimation, truncated exponential backoff and retry
quota accounting that drive adaptive and standard retry mode.

Installable alongside the real `aiobotocore`, which the parity tests compare
against directly.
"""

from ._lib import (
    BUCKET_ACQUIRE,
    BUCKET_MIN_RATE,
    BUCKET_SET_RATE,
    BUCKET_TICK,
    CRC32_POLY,
    CRC32C_POLY,
    QUOTA_ACQUIRE,
    QUOTA_RELEASE,
    aws_chunked_encode,
    backoff_delays,
    crc32,
    crc32c,
    cubic_error,
    cubic_success,
    cubic_zero_point,
    hex_len,
    rate_clock,
    retry_quota_trace,
    token_bucket_trace,
)

__all__ = [
    "aws_chunked_encode",
    "backoff_delays",
    "crc32",
    "crc32c",
    "cubic_error",
    "cubic_success",
    "cubic_zero_point",
    "hex_len",
    "rate_clock",
    "retry_quota_trace",
    "token_bucket_trace",
]
__version__ = "0.1.0"
