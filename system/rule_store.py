"""
rule_store.py — fixed-point encoding and the deduplication-parameter
rule store. This is the core of PrivPathInfer-static.

The rule store holds one EncryptedRule per condition occurrence across
all root-to-leaf paths of the tree. A single tunable parameter, c,
controls how many occurrences of the same (feature_index, threshold)
value pair share one Paillier ciphertext:

    c = 1        every occurrence gets its own fresh ciphertext
                 (maximum independence). Feature identity is concealed
                 behind a PRP tag, since there is nothing to link.

    c > 1        occurrences of the same (feature_index, threshold)
                 pair are grouped into buckets of at most c and share
                 one ciphertext per bucket. Feature identity is stored
                 in the clear instead of PRP-tagged: concealment and
                 sharing-based deduplication are in tension, because
                 detecting which rules share a ciphertext requires the
                 tag to be either PRP-concealed with no sharing, or
                 shared in the clear — not both at once.

    c = "max"    every occurrence of a given (feature_index, threshold)
                 pair shares a single ciphertext (one bucket per pair).

This is a single continuum parameterized by c, not two named "modes".
"""

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Union

from crypto import paillier
from crypto.prf_prp import prp_int

SCALE_FACTOR = 10000
OFFSET = 10 ** 9


def encode(value: float) -> int:
    """Fixed-point encoding: truncates, does not round."""
    return int(value * SCALE_FACTOR) + OFFSET


def decode(encoded: int) -> float:
    """Inverse of encode(). Testing/debugging only — never used in the
    actual protocol."""
    return (encoded - OFFSET) / SCALE_FACTOR


@dataclass(frozen=True)
class EncryptedRule:
    feature_tag: int      # PRP-permuted feature index, or plaintext feature index
    threshold_ct: int     # Paillier ciphertext of the encoded threshold
    direction: str        # "left" or "right"
    path_id: int
    label: int


@dataclass
class RuleStore:
    rules: List[EncryptedRule]
    public_key: Tuple[int, int]
    conceal_feature: bool


# Serialized byte-size model used by measure_storage_bytes().
FEATURE_TAG_CONCEALED_BYTES = 16   # AES-128/PRP output width
FEATURE_TAG_PLAIN_BYTES = 4        # plaintext feature index (int32)
DIRECTION_BYTES = 1
PATH_ID_BYTES = 4
LABEL_BYTES = 1


def build_rule_store(paths, public_key, c: Union[int, str], prp_key: bytes = None) -> RuleStore:
    """
    Build the rule store for the given extracted paths at deduplication
    parameter c.

    Args:
        paths:      list of system.path_extractor.Path
        public_key: Paillier (n, g)
        c:          int >= 1, or the string "max" (one ciphertext per
                    distinct (feature_index, threshold) pair, regardless
                    of how many times it occurs)
        prp_key:    16-byte key for feature-tag concealment, only used
                    when c == 1. Generated if not supplied.

    Returns:
        (RuleStore, prp_key): prp_key is the key actually used for
        feature-tag concealment (None when c != 1). It is NOT part of
        the RuleStore object the Cloud holds — the caller must pass it
        to UserParty separately so query tags match the store's tags.
    """
    conceal_feature = (c == 1)
    if conceal_feature and prp_key is None:
        prp_key = os.urandom(16)

    groups = {}
    for path in paths:
        for cond in path.conditions:
            key = (cond.feature_index, cond.threshold)
            groups.setdefault(key, []).append((cond.direction, path.path_id, path.label))

    rules: List[EncryptedRule] = []

    for (feature_index, threshold), occurrences in groups.items():
        bucket_size = len(occurrences) if c == "max" else c
        encoded_threshold = encode(threshold)

        feature_tag = prp_int(prp_key, feature_index) if conceal_feature else feature_index

        for start in range(0, len(occurrences), bucket_size):
            bucket = occurrences[start:start + bucket_size]
            ct = paillier.encrypt(encoded_threshold, public_key)
            for direction, path_id, label in bucket:
                rules.append(EncryptedRule(
                    feature_tag=feature_tag,
                    threshold_ct=ct,
                    direction=direction,
                    path_id=path_id,
                    label=label,
                ))

    store = RuleStore(rules=rules, public_key=public_key, conceal_feature=conceal_feature)
    return store, (prp_key if conceal_feature else None)


def distinct_ciphertext_count(rule_store: RuleStore) -> int:
    """Number of distinct ciphertexts backing the rule store."""
    return len({r.threshold_ct for r in rule_store.rules})


def linkability(rule_store: RuleStore) -> int:
    """Maximum number of rules referencing the same ciphertext."""
    if not rule_store.rules:
        return 0
    counts = Counter(r.threshold_ct for r in rule_store.rules)
    return max(counts.values())


def measure_storage_bytes(rule_store: RuleStore) -> int:
    """
    Total serialized byte size of the rule store: one copy of each
    distinct ciphertext, plus per-rule bookkeeping (feature tag,
    direction, path_id, label).
    """
    n, _g = rule_store.public_key
    ciphertext_byte_size = (n.bit_length() * 2 + 7) // 8  # ciphertexts live in [0, n^2)

    n_ciphertexts = distinct_ciphertext_count(rule_store)
    tag_bytes = FEATURE_TAG_CONCEALED_BYTES if rule_store.conceal_feature else FEATURE_TAG_PLAIN_BYTES
    bookkeeping_per_rule = tag_bytes + DIRECTION_BYTES + PATH_ID_BYTES + LABEL_BYTES

    return n_ciphertexts * ciphertext_byte_size + len(rule_store.rules) * bookkeeping_per_rule
