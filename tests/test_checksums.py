"""Parity for the checksum and aws-chunked framing kernels.

CRC-32 is compared against `zlib.crc32` and against
`botocore.httpchecksum.Crc32Checksum`, the accumulator aiobotocore's
`StreamingChecksumBody` actually updates. CRC-32C is compared against the
`crc32c` package, which is the reference implementation botocore's CRT
checksum class delegates to. Both are exact integer algorithms, so the
assertions are exact equality.
"""

import zlib

import numpy as np
import pytest

import mojo_aiobotocore as mab


def _rand(n, seed):
    return bytes(
        np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tolist()
    )


# ---------------------------------------------------------------------------
# CRC tables
# ---------------------------------------------------------------------------


def _python_crc_table(poly: int) -> list:
    """The standard reflected CRC table, built independently of the kernel."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc)
    return table


def test_crc32_table_matches_the_standard_table():
    from mojo_aiobotocore._lib import lib, CRC32_POLY

    table = np.zeros(256, dtype=np.uint32)
    lib.ab_crc_table(table.ctypes.data, CRC32_POLY)
    assert [int(x) for x in table] == _python_crc_table(CRC32_POLY)


def test_crc32c_table_matches_the_standard_table():
    from mojo_aiobotocore._lib import lib, CRC32C_POLY

    table = np.zeros(256, dtype=np.uint32)
    lib.ab_crc_table(table.ctypes.data, CRC32C_POLY)
    assert [int(x) for x in table] == _python_crc_table(CRC32C_POLY)


def test_table_entry_relates_to_the_single_byte_checksum():
    """table[i] is the intermediate value, not the finished digest: pin the
    relation so a table built with the wrong polynomial cannot pass."""
    import crc32c

    from mojo_aiobotocore._lib import lib, CRC32_POLY, CRC32C_POLY

    t32 = np.zeros(256, dtype=np.uint32)
    lib.ab_crc_table(t32.ctypes.data, CRC32_POLY)
    tc = np.zeros(256, dtype=np.uint32)
    lib.ab_crc_table(tc.ctypes.data, CRC32C_POLY)
    for b in (0, 1, 63, 127, 128, 200, 255):
        want32 = (~(int(t32[(255 - b) & 0xFF]) ^ (0xFFFFFFFF >> 8))) & 0xFFFFFFFF
        wantc = (~(int(tc[(255 - b) & 0xFF]) ^ (0xFFFFFFFF >> 8))) & 0xFFFFFFFF
        assert want32 == zlib.crc32(bytes([b]))
        assert wantc == crc32c.crc32c(bytes([b]))


# ---------------------------------------------------------------------------
# CRC-32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 4096, 65537])
def test_crc32_matches_zlib(n):
    data = _rand(n, n)
    assert mab.crc32(data) == zlib.crc32(data)


def test_crc32_known_vector():
    assert mab.crc32(b"123456789") == 0xCBF43926


def test_crc32_is_streaming_in_chunks():
    """A kernel that reset its seed per call would agree on one shot and
    disagree on a chunked update, which is exactly how
    `Crc32Checksum.update` is called."""
    from botocore.httpchecksum import Crc32Checksum

    data = _rand(10000, 3)
    checksum = Crc32Checksum()
    seed = 0
    for i in range(0, len(data), 137):
        chunk = data[i : i + 137]
        seed = mab.crc32(chunk, seed)
        checksum.update(chunk)
    assert seed == int.from_bytes(checksum.digest(), "big")
    assert seed == zlib.crc32(data)


def test_crc32_seed_survives_an_empty_chunk():
    from botocore.httpchecksum import Crc32Checksum

    data = _rand(5000, 4)
    checksum = Crc32Checksum()
    seed = 0
    for i in range(0, len(data), 499):
        seed = mab.crc32(data[i : i + 499], seed)
        checksum.update(data[i : i + 499])
        seed = mab.crc32(b"", seed)
        checksum.update(b"")
    assert seed == int.from_bytes(checksum.digest(), "big")


# ---------------------------------------------------------------------------
# CRC-32C
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 4096, 65537])
def test_crc32c_matches_the_crc32c_package(n):
    import crc32c

    data = _rand(n, n + 100)
    assert mab.crc32c(data) == crc32c.crc32c(data)


def test_crc32c_known_vector():
    assert mab.crc32c(b"123456789") == 0xE3069283


def test_crc32c_differs_from_crc32():
    data = b"123456789"
    assert mab.crc32(data) != mab.crc32c(data)


def test_crc32c_is_streaming_in_chunks():
    import crc32c

    data = _rand(9000, 5)
    seed = 0
    for i in range(0, len(data), 91):
        seed = mab.crc32c(data[i : i + 91], seed)
    assert seed == crc32c.crc32c(data)


# ---------------------------------------------------------------------------
# aws-chunked framing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expect",
    [
        (0, b"0"),
        (1, b"1"),
        (15, b"f"),
        (16, b"10"),
        (255, b"ff"),
        (256, b"100"),
        (4096, b"1000"),
        (8191, b"1fff"),
        (8192, b"2000"),
        (1 << 40, b"10000000000"),
    ],
)
def test_hex_len_matches_pythons_hex(n, expect):
    assert mab.hex_len(n) == hex(n)[2:].encode("ascii")


def test_hex_len_has_no_leading_zeros():
    assert not mab.hex_len(1024).startswith(b"0")


def test_aws_chunked_framing_matches_the_wrapper():
    """Drive aiobotocore's own `AioAwsChunkedWrapper` and compare the bytes it
    emits with the kernel's framing."""
    import asyncio

    from aiobotocore.httpchecksum import AioAwsChunkedWrapper

    class _Raw:
        def __init__(self, data, size):
            self._data = data
            self._size = size

        async def read(self, amt=None):
            out = self._data[: self._size]
            self._data = self._data[self._size :]
            return out

    async def collect(data, chunk_size):
        wrapper = AioAwsChunkedWrapper(_Raw(data, chunk_size), chunk_size=chunk_size)
        out = b""
        async for piece in wrapper:
            out += piece
        return out

    for payload, chunk in [
        (b"", 8),
        (b"a", 8),
        (b"abcdefghij", 8),
        (_rand(1000, 7), 64),
        (_rand(1000, 8), 1000),
    ]:
        got, _crc = mab.aws_chunked_encode(payload, chunk_size=chunk)
        expect = asyncio.run(collect(payload, chunk))
        assert got == expect


def test_aws_chunked_crc_is_the_crc_of_the_plaintext():
    for payload, chunk in [(b"", 8), (b"hello", 8), (_rand(3000, 9), 128)]:
        _body, crc = mab.aws_chunked_encode(payload, chunk_size=chunk)
        assert crc == zlib.crc32(payload)


def test_aws_chunked_decodes_back_to_the_input():
    payload = _rand(5000, 10)
    body, _crc = mab.aws_chunked_encode(payload, chunk_size=97)
    out = bytearray()
    pos = 0
    while True:
        nl = body.index(b"\r\n", pos)
        size = int(body[pos:nl], 16)
        pos = nl + 2
        if size == 0:
            break
        out += body[pos : pos + size]
        pos += size + 2
    assert bytes(out) == payload


def test_aws_chunked_rejects_a_zero_chunk_size():
    with pytest.raises(ValueError):
        mab.aws_chunked_encode(b"abc", chunk_size=0)
