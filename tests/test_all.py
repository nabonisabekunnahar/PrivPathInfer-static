"""
test_all.py — test suite for privpathinfer-static.

Covers: fixed-point encoding (round-trip and the truncation boundary),
Paillier correctness and homomorphism, the deduplication rule-store
construction at several values of c, the two-round secure comparison
protocol against plaintext classification, and the subtree partitioner
(baseline vs router-depth classification) against plaintext.

Usage: python -m tests.test_all
"""

import os
import sys
import random
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from crypto.prf_prp import prp_int
from system.path_extractor import extract_paths
from system.rule_store import (
    encode, decode, build_rule_store, distinct_ciphertext_count, linkability, RuleStore,
)
from system.inference_engine import CloudParty, UserParty
from system.subtree_partitioner import (
    build_router_map, build_partitioned_rules, baseline_classify, subtree_classify,
)


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, func):
        try:
            func()
            self.passed += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  [ERROR] {name}")
            print(traceback.format_exc())

    def summary(self):
        total = self.passed + self.failed
        print(f"\nTOTAL {self.passed}/{total} TESTS PASSED")
        return self.failed == 0


TEST_BITS = 512  # unit tests only care about correctness, not the reported timings


def _small_tree():
    """A tiny deterministic synthetic tree with 2 features, depth 3."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, size=(200, 2))
    y = ((X[:, 0] > 5) & (X[:, 1] > 3)).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    return clf, X, y


# ---------------------------------------------------------------------------
# Fixed-point encoding
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    for v in [0.0, 1.2345, -3.4567, 126.5432, 9999.9999]:
        assert abs(decode(encode(v)) - v) < 1e-4, v


def test_encode_truncation_boundary():
    # Values differing only past the 4th decimal place must encode identically.
    a, b = 1.23456, 1.23459
    assert encode(a) == encode(b)
    # But a difference at the 4th decimal place must NOT collide.
    c, d = 1.2345, 1.2346
    assert encode(c) != encode(d)


# ---------------------------------------------------------------------------
# Paillier correctness
# ---------------------------------------------------------------------------

def test_paillier_encrypt_decrypt():
    pub, priv = paillier.keygen(bits=TEST_BITS)
    for _ in range(10):
        m = random.randrange(0, pub[0])
        c = paillier.encrypt(m, pub)
        assert paillier.decrypt(c, pub, priv) == m


def test_paillier_homomorphic_add_subtract():
    pub, priv = paillier.keygen(bits=TEST_BITS)
    n = pub[0]
    m1, m2 = random.randrange(1, n // 4), random.randrange(1, n // 4)
    c1, c2 = paillier.encrypt(m1, pub), paillier.encrypt(m2, pub)

    c_sum = paillier.add_encrypted(c1, c2, pub)
    assert paillier.decrypt(c_sum, pub, priv) == (m1 + m2) % n

    c_diff = paillier.subtract_encrypted(c1, c2, pub)
    assert paillier.decrypt(c_diff, pub, priv) == (m1 - m2) % n


# ---------------------------------------------------------------------------
# Deduplication rule-store construction
# ---------------------------------------------------------------------------

def test_dedup_ciphertext_count_and_linkability():
    clf, X, y = _small_tree()
    paths = extract_paths(clf)
    pub, priv = paillier.keygen(bits=TEST_BITS)

    # Total occurrences across all paths, grouped by (feature, threshold).
    groups = {}
    for path in paths:
        for cond in path.conditions:
            key = (cond.feature_index, cond.threshold)
            groups.setdefault(key, []).append(cond)
    total_occurrences = sum(len(v) for v in groups.values())

    # c = 1: every occurrence gets its own ciphertext, linkability 1.
    store_c1, _ = build_rule_store(paths, pub, c=1)
    assert distinct_ciphertext_count(store_c1) == total_occurrences
    assert linkability(store_c1) == 1
    assert store_c1.conceal_feature is True

    # c = "max": one ciphertext per distinct (feature, threshold) pair.
    store_max, _ = build_rule_store(paths, pub, c="max")
    assert distinct_ciphertext_count(store_max) == len(groups)
    assert linkability(store_max) == max(len(v) for v in groups.values())
    assert store_max.conceal_feature is False

    # c = 2: bucket size at most 2 per (feature, threshold) group.
    store_c2, _ = build_rule_store(paths, pub, c=2)
    expected_buckets = sum(-(-len(v) // 2) for v in groups.values())  # ceil division
    assert distinct_ciphertext_count(store_c2) == expected_buckets
    assert linkability(store_c2) <= 2

    # Row count (number of EncryptedRule entries) must equal the total
    # occurrence count regardless of c — sharing affects ciphertexts,
    # not row count.
    assert len(store_c1.rules) == total_occurrences
    assert len(store_max.rules) == total_occurrences
    assert len(store_c2.rules) == total_occurrences


# ---------------------------------------------------------------------------
# Two-round protocol vs plaintext
# ---------------------------------------------------------------------------

def test_protocol_matches_plaintext():
    clf, X, y = _small_tree()
    paths = extract_paths(clf)
    pub, priv = paillier.keygen(bits=TEST_BITS)

    store, prp_key = build_rule_store(paths, pub, c=1)
    cloud = CloudParty(pub, store)
    user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=store.conceal_feature)

    rng = np.random.RandomState(1)
    queries = rng.uniform(0, 10, size=(15, 2))
    for row in queries:
        expected = int(clf.predict([row])[0])
        got = user.classify(cloud, row.tolist(), paths)
        assert got == expected, (row, expected, got)


# ---------------------------------------------------------------------------
# Subtree partitioner vs plaintext, and vs the unpartitioned baseline
# ---------------------------------------------------------------------------

def test_subtree_matches_plaintext_and_baseline():
    clf, X, y = _small_tree()
    paths = extract_paths(clf)
    pub, priv = paillier.keygen(bits=TEST_BITS)

    baseline_store, prp_key = build_rule_store(paths, pub, c=1)
    user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=True)

    router_depth = 1
    router_map = build_router_map(paths, router_depth)
    sub_rules, router_positions, subtree_positions = build_partitioned_rules(paths, router_depth, pub, prp_key)

    rng = np.random.RandomState(2)
    queries = rng.uniform(0, 10, size=(15, 2))
    for row in queries:
        expected = int(clf.predict([row])[0])

        tagged_b = user.tag_feature_vector(row.tolist())
        cloud_b = CloudParty(pub, baseline_store)
        blinded_b = cloud_b.evaluate_round(tagged_b)
        got_baseline = baseline_classify(user, baseline_store.rules, blinded_b, paths)

        cloud_s = CloudParty(pub, RuleStore(rules=sub_rules, public_key=pub, conceal_feature=True))
        tagged_s = user.tag_feature_vector(row.tolist())
        blinded_s = cloud_s.evaluate_round(tagged_s)
        got_subtree = subtree_classify(user, sub_rules, blinded_s, paths, router_map, router_positions, subtree_positions)

        assert got_baseline == expected, (row, expected, got_baseline)
        assert got_subtree == expected, (row, expected, got_subtree)
        assert got_subtree == got_baseline


# ---------------------------------------------------------------------------
# Partitioned store must be indistinguishable from the baseline to the Cloud
# ---------------------------------------------------------------------------

def test_partitioned_store_indistinguishable_from_baseline():
    clf, X, y = _small_tree()
    paths = extract_paths(clf)
    pub, priv = paillier.keygen(bits=TEST_BITS)

    baseline_store, prp_key = build_rule_store(paths, pub, c=1)

    raw_feature_indices = {cond.feature_index for path in paths for cond in path.conditions}
    expected_tags = {prp_int(prp_key, i) for i in raw_feature_indices}

    for router_depth in (1, 2):
        sub_rules, _router_positions, _subtree_positions = build_partitioned_rules(
            paths, router_depth, pub, prp_key
        )

        # 1. Rule count must equal the baseline's — otherwise the Cloud
        # can infer the router depth from the store's size alone.
        assert len(sub_rules) == len(baseline_store.rules), (
            len(sub_rules), len(baseline_store.rules), router_depth
        )

        # 2. Every rule (router, subtree, or padding) must carry a
        # real, non-negative path_id and label — a -1 sentinel would
        # mark a rule as a router row.
        for rule in sub_rules:
            assert rule.path_id >= 0, (rule, router_depth)
            assert rule.label >= 0, (rule, router_depth)

        # 3. No rule may carry a raw feature index — every feature_tag
        # must be a PRP output matching the baseline's own tagging.
        for rule in sub_rules:
            assert rule.feature_tag not in raw_feature_indices, (rule, router_depth)
            assert rule.feature_tag in expected_tags, (rule, router_depth)


def main():
    runner = TestRunner()
    tests = [
        ("encode/decode round-trip", test_encode_decode_roundtrip),
        ("encode truncation boundary", test_encode_truncation_boundary),
        ("Paillier encrypt/decrypt", test_paillier_encrypt_decrypt),
        ("Paillier homomorphic add/subtract", test_paillier_homomorphic_add_subtract),
        ("dedup ciphertext count & linkability", test_dedup_ciphertext_count_and_linkability),
        ("two-round protocol matches plaintext", test_protocol_matches_plaintext),
        ("subtree partitioner matches plaintext & baseline", test_subtree_matches_plaintext_and_baseline),
        ("partitioned store indistinguishable from baseline", test_partitioned_store_indistinguishable_from_baseline),
    ]
    for name, func in tests:
        runner.run(name, func)
    ok = runner.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
