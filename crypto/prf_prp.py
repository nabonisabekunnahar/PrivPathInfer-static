"""
prf_prp.py — Pseudorandom Function (PRF) and Pseudorandom Permutation (PRP)
============================================================================
PrivPathInfer: Privacy-Preserving Decision Tree Inference Framework

Implementation Reference:
    Boneh, D. and Shoup, V. "A Graduate Course in Applied Cryptography"
    Version 0.5, 2020. Chapter 4, Sections 4.1 and 4.4.
    https://crypto.stanford.edu/~dabo/cryptobook/

Security Assumptions:
    PRF Security (Definition 4.2, Boneh-Shoup):
        F : K × X → Y is a secure PRF if for all efficient adversaries A:
            PRFadv[A, F] = |Pr[A^{F(K,·)} = 1] - Pr[A^{f(·)} = 1]| ≤ negl(λ)
        where f is a truly random function from X to Y.

    PRP Security (Definition 4.1, Boneh-Shoup):
        AES-128 is modeled as a secure pseudorandom permutation (PRP)
        over domain {0,1}^128. It is bijective: different inputs always
        produce different outputs under the same key.

    PRF/PRP Switching Lemma (Boneh-Shoup, Section 4.4):
        For AES-128 with domain size |X| = 2^128, a PRP is
        computationally indistinguishable from a PRF. The distinguishing
        advantage is bounded by Q^2 / 2^128, which is negligible for
        any practical Q. Therefore, AES-128 is used as the PRF
        instantiation throughout PrivPathInfer.

Usage in PrivPathInfer:
    PRP: feature-index permutation in the rule store — the PRP-permuted
         tag conceals which plaintext feature a rule's threshold belongs
         to, so the cloud cannot infer feature identity by index alone.

Author: Mst Sabekunnahar Naboni (Roll: 2007034)
        BSc in CSE, KUET
        Thesis: CSE 4000
"""

import os
import struct
from crypto.aes128 import aes_encrypt, aes_decrypt


# ---------------------------------------------------------------------------
# Helper: encode inputs to 16-byte AES blocks
# ---------------------------------------------------------------------------

def _encode_to_block(value, block_size=16):
    """
    Encode an integer or bytes value into a fixed-size 16-byte block.

    For integers: encode as big-endian, zero-padded to block_size bytes.
    For bytes: zero-pad or truncate to block_size bytes.
    For strings: UTF-8 encode then pad/truncate.

    Args:
        value:      int, bytes, or str to encode
        block_size: output size in bytes (default 16 for AES-128)

    Returns:
        bytes of length block_size
    """
    if isinstance(value, int):
        # Encode integer as big-endian bytes, padded to block_size
        byte_length = (value.bit_length() + 7) // 8
        byte_length = max(byte_length, 1)
        raw = value.to_bytes(byte_length, byteorder='big')
    elif isinstance(value, str):
        raw = value.encode('utf-8')
    elif isinstance(value, bytes):
        raw = value
    else:
        raise TypeError(f"Unsupported input type: {type(value)}")

    if len(raw) > block_size:
        # Truncate from the right (keep high-order bytes)
        raw = raw[:block_size]
    else:
        # Zero-pad on the left
        raw = raw.rjust(block_size, b'\x00')

    return raw


def _encode_key(key, key_size=16):
    """
    Encode a key value into a fixed-size 16-byte AES key.

    Args:
        key:      int, bytes, or str
        key_size: must be 16 for AES-128

    Returns:
        bytes of length 16
    """
    return _encode_to_block(key, key_size)


# ---------------------------------------------------------------------------
# PRF: Pseudorandom Function
# Reference: Boneh-Shoup, Definition 4.2 and Section 4.4
# ---------------------------------------------------------------------------

def prf(key, x):
    """
    Pseudorandom Function: F(key, x) = AES_key(x)

    Instantiates the PRF using AES-128 as the underlying PRP.
    By the PRF/PRP switching lemma (Boneh-Shoup, Section 4.4),
    AES-128 is computationally indistinguishable from a truly random
    function over domain {0,1}^128.

    Security Guarantee:
        For any efficient adversary A making at most Q oracle queries:
            PRFadv[A, F] ≤ Q^2 / 2^128  (negligible)

    Args:
        key: secret key — int, bytes (16), or str
             Conceptually: key ∈ K = {0,1}^128
        x:   input — int, bytes (≤16), or str
             Conceptually: x ∈ X = {0,1}^128

    Returns:
        bytes of length 16: the PRF output y ∈ Y = {0,1}^128
    """
    k = _encode_key(key)
    block = _encode_to_block(x)
    return aes_encrypt(block, k)


def prf_int(key, x):
    """
    PRF with integer output.

    Returns the PRF output as a Python integer (big-endian interpretation
    of the 16-byte AES output).

    Args:
        key: secret key (int, bytes, or str)
        x:   input (int, bytes, or str)

    Returns:
        int: 128-bit PRF output
    """
    output_bytes = prf(key, x)
    return int.from_bytes(output_bytes, byteorder='big')


# ---------------------------------------------------------------------------
# PRP: Pseudorandom Permutation
# Reference: Boneh-Shoup, Definition 4.1
# ---------------------------------------------------------------------------

def prp(key, x):
    """
    Pseudorandom Permutation: P(key, x) = AES_key(x)

    AES-128 is directly a PRP: it is bijective (different inputs always
    produce different outputs under the same key) and computationally
    indistinguishable from a random permutation.

    Security Guarantee (Definition 4.1, Boneh-Shoup):
        For any efficient adversary A:
            PRPadv[A, AES] ≤ negl(λ)

    Bijection Property:
        For all x₁ ≠ x₂ and fixed key k:
            PRP(k, x₁) ≠ PRP(k, x₂)   (no collisions)

    Usage in PrivPathInfer:
        Feature-index tagging in the rule store (Theorem 1):
            feature_tag = prp(permutation_key, feature_index)
        Prevents the cloud from inferring feature identity from the tag.

    Args:
        key: secret key — int, bytes (16), or str
        x:   input — int, bytes (≤16), or str

    Returns:
        bytes of length 16: the PRP output
    """
    k = _encode_key(key)
    block = _encode_to_block(x)
    return aes_encrypt(block, k)


def prp_inverse(key, y):
    """
    Inverse PRP: P^{-1}(key, y) = AES_key^{-1}(y)

    Since AES is a permutation, it has a well-defined inverse.
    The inverse is used during decryption to recover original indices.

    Args:
        key: secret key — int, bytes (16), or str
        y:   PRP output — bytes (16)

    Returns:
        bytes of length 16: the original input x such that prp(key, x) = y
    """
    k = _encode_key(key)
    block = _encode_to_block(y)
    return aes_decrypt(block, k)


def prp_int(key, x):
    """
    PRP with integer output.

    Args:
        key: secret key (int, bytes, or str)
        x:   input (int, bytes, or str)

    Returns:
        int: 128-bit PRP output
    """
    output_bytes = prp(key, x)
    return int.from_bytes(output_bytes, byteorder='big')


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """
    Verify PRF and PRP properties required for PrivPathInfer security.

    Tests:
        1. PRF determinism: same (key, x) always gives same output
        2. PRF pseudorandomness proxy: different keys give different outputs
        3. PRF input sensitivity: different inputs give different outputs
        4. PRP bijectivity: different inputs give different outputs (no collision)
        5. PRP invertibility: prp_inverse(prp(x)) == x
        6. Integer interface: prf_int returns a valid 128-bit integer
    """

    print("=" * 60)
    print("PRF/PRP Verification Tests")
    print("Reference: Boneh-Shoup, Definitions 4.1 and 4.2")
    print("=" * 60)

    key1 = os.urandom(16)
    key2 = os.urandom(16)
    x1   = os.urandom(16)
    x2   = os.urandom(16)

    # Test 1: PRF determinism
    assert prf(key1, x1) == prf(key1, x1), "PRF must be deterministic"
    print("[PASS] Test 1: PRF determinism — same (key, x) → same output")

    # Test 2: PRF key sensitivity
    out1 = prf(key1, x1)
    out2 = prf(key2, x1)
    assert out1 != out2, "Different keys must (almost certainly) give different PRF outputs"
    print("[PASS] Test 2: PRF key sensitivity — different keys → different outputs")

    # Test 3: PRF input sensitivity
    out3 = prf(key1, x1)
    out4 = prf(key1, x2)
    assert out3 != out4, "Different inputs must (almost certainly) give different PRF outputs"
    print("[PASS] Test 3: PRF input sensitivity — different inputs → different outputs")

    # Test 4: PRP bijectivity (no collision under same key)
    # For AES-128 with 2^128 domain, collisions are negligible
    p1 = prp(key1, x1)
    p2 = prp(key1, x2)
    assert p1 != p2, "PRP must be bijective: different inputs → different outputs"
    print("[PASS] Test 4: PRP bijectivity — different inputs → different outputs")

    # Test 5: PRP invertibility
    y = prp(key1, x1)
    recovered = prp_inverse(key1, y)
    assert _encode_to_block(x1) == recovered, "PRP inverse must recover original input"
    print("[PASS] Test 5: PRP invertibility — prp_inverse(prp(x)) == x")

    # Test 6: Integer interface
    prf_val = prf_int(key1, x1)
    assert isinstance(prf_val, int), "prf_int must return an integer"
    assert 0 <= prf_val < 2**128, "PRF output must be a 128-bit integer"
    print("[PASS] Test 6: Integer interface — prf_int returns valid 128-bit integer")

    print("\n[ALL TESTS PASSED] prf_prp.py is verified and ready.")
    print("Security: PRF/PRP instantiated with AES-128.")
    print("Reference: Boneh-Shoup, Section 4.4 (PRF/PRP Switching Lemma)")


if __name__ == "__main__":
    run_all_tests()
