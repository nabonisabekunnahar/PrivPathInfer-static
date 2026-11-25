"""
test_edge_cases.py - unit tests added for the camera-ready revision.

Camera-ready comments addressed: R3-11 (equality at thresholds, overflow
bounds, range of the blinding factor), R3-12 (exact message count),
R3-9 (key ownership), R3-6 (negative / extreme inputs, wraparound).

Usage: python -m tests.test_edge_cases        (about a minute at 512-bit keys)
"""

import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import encode, OFFSET, SCALE_FACTOR, build_rule_store
from system.inference_engine import CloudParty, UserParty
from experiments.datasets import load_dataset

R_MAX = 2 ** 64
PUB, PRIV = paillier.keygen(bits=512)
USER = UserParty(PUB, PRIV)


def sign_with(x_enc, th_enc, r, pub=PUB, priv=PRIV, user=USER):
    """Direction the user recovers for one condition with a chosen blinding r."""
    diff = paillier.subtract_encrypted(paillier.encrypt(x_enc, pub), paillier.encrypt(th_enc, pub), pub)
    return user._recover_direction(paillier.scalar_multiply(diff, r, pub))


def test_equality_goes_left():
    for r in (1, 2, R_MAX - 1):
        assert sign_with(OFFSET + 5000, OFFSET + 5000, r) == "left"   # x == theta -> "left", like x <= theta


def test_blinding_factor_boundaries_keep_the_sign():
    for r in (1, R_MAX - 1):
        assert sign_with(OFFSET + 1, OFFSET, r) == "right"
        assert sign_with(OFFSET - 1, OFFSET, r) == "left"


def test_encoding_is_truncation_toward_zero():
    assert encode(1.2345) == OFFSET + 12345
    assert encode(1.23459) == encode(1.2345)          # past the 4th decimal: same integer
    assert encode(-0.00005) == OFFSET                  # toward zero, not floor (floor would give OFFSET - 1)


def test_inputs_below_range_are_rejected():
    assert encode(-1e5) == 0                           # lowest representable value
    try:
        paillier.encrypt(encode(-1e5 - 1), PUB)
    except Exception:
        return
    raise AssertionError("a negative plaintext was accepted by encrypt()")


def test_precision_bound_is_real():
    """x slightly above theta inside the same 1e-4 cell compares as equal: the
    stated precision bound. One full unit above, the protocol separates them."""
    theta = 126.5
    assert encode(theta + 0.00005) == encode(theta)    # same encoded integer
    assert sign_with(encode(theta + 0.00005), encode(theta), 7) == "left"   # plaintext would say "right"
    assert sign_with(encode(theta + 0.0001), encode(theta), 7) == "right"


def test_wraparound_flips_sign_only_above_the_bound():
    pub, priv = paillier.keygen(bits=128)
    user = UserParty(pub, priv)
    n = pub[0]
    safe = (n // 2) // (R_MAX - 1)
    wrong_below = sum(sign_with(OFFSET + safe // 4, OFFSET, random.randrange(1, R_MAX), pub, priv, user) != "right"
                      for _ in range(100))
    wrong_above = sum(sign_with(OFFSET + 2 ** 70, OFFSET, random.randrange(1, R_MAX), pub, priv, user) != "right"
                      for _ in range(100))
    assert wrong_below == 0, wrong_below
    assert wrong_above > 0, "expected some sign flips far above the bound"


def test_other_users_key_cannot_decrypt():
    pub_b, priv_b = paillier.keygen(bits=512)
    c = paillier.encrypt(5, PUB)
    try:
        m = paillier.decrypt(c, pub_b, priv_b)
    except Exception:
        return
    assert m != 5, "a ciphertext under one user's key decrypted correctly under another's"


def test_one_query_one_response():
    """classify() sends exactly one encrypted query and receives exactly one
    response: two messages, one round trip."""
    X, y = load_dataset("pima", data_dir="data")
    clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    paths = extract_paths(clf)
    store, prp_key = build_rule_store(paths, PUB, c=1)

    class Counting(CloudParty):
        __slots__ = ("calls",)

        def evaluate_round(self, tagged, rng=None):
            self.calls = getattr(self, "calls", 0) + 1
            return super().evaluate_round(tagged, rng)

    cloud = Counting(PUB, store)
    user = UserParty(PUB, PRIV, prp_key=prp_key, conceal_feature=store.conceal_feature)
    sent = []
    original = user.tag_feature_vector
    user.tag_feature_vector = lambda v: (sent.append(1), original(v))[1]
    label = user.classify(cloud, X[0].tolist(), paths)
    assert label == int(clf.predict(X[:1])[0])
    assert cloud.calls == 1 and len(sent) == 1


def test_cloud_cannot_hold_a_private_key():
    store, _ = build_rule_store(extract_paths(DecisionTreeClassifier(max_depth=2, random_state=42).fit(
        *load_dataset("pima", data_dir="data"))), PUB, c=1)
    cloud = CloudParty(PUB, store)
    assert not hasattr(cloud, "decrypt")
    try:
        cloud.private_key = PRIV
    except AttributeError:
        return
    raise AssertionError("CloudParty accepted a private key attribute")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception:
            failed += 1
            print(f"[FAIL] {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
