from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _shingles(text: str, size: int = 3) -> Iterable[str]:
    words = text.lower().split()
    if len(words) < size:
        return (" ".join(words),) if words else ()
    return (" ".join(words[index:index + size]) for index in range(len(words) - size + 1))


def simhash(text: str, bits: int = 64) -> int:
    """Return a deterministic SimHash fingerprint for normalized text."""
    weights = [0] * bits
    for shingle in _shingles(text):
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        value = int.from_bytes(digest[: bits // 8], "big")
        for bit in range(bits):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two fingerprints."""
    return (left ^ right).bit_count()


def is_near_duplicate(
    fingerprint: int,
    previous_fingerprints: Iterable[int],
    max_hamming_distance: int = 6,
) -> bool:
    """Detect highly similar documents using SimHash distance."""
    return any(
        hamming_distance(fingerprint, previous) <= max_hamming_distance
        for previous in previous_fingerprints
    )
