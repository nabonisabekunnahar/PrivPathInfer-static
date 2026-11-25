"""
exp_edge_cases.py - numerical edge cases of the fixed-point encoding and of
sign recovery under blinding.

Camera-ready comments addressed: R3-6 (values near thresholds, negative and
extreme values, precision changes, modular wraparound) and R3-11 (equality at
thresholds, overflow bounds, range of the blinding factor).

    python -m experiments.exp_edge_cases              # 1024-bit crypto checks
    python -m experiments.exp_edge_cases --bits 512   # faster

Parts
  1  encoding audit: float artefacts on the 4-decimal grid, and how much of
     each dataset has more than 4 decimals (so 4-decimal truncation merges
     distinct raw values)
  2  threshold-adjacent values, three ways: plaintext x <= theta, the integer
     comparison of the encodings, and the real Paillier protocol
  3  negative and extreme inputs
  4  modular wraparound: safe bound per key size, and a demonstration that
     the sign flips once |x_enc - theta_enc| * r exceeds n/2 (128-bit demo key)
  5  precision sweep (scale 10^2, 10^4, 10^6, 10^8): encoding, comparison and
     classification mismatch counts, integer comparison only. This is
     the same comparison the protocol makes whenever no wraparound occurs
     (Part 2 confirms this against the real protocol)

Output: results/exp_edge_cases.json
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.rule_store import encode, OFFSET, SCALE_FACTOR
from system.inference_engine import UserParty
from experiments.datasets import load_dataset, DATASET_NAMES
from experiments._meta import run_meta

R_MAX = 2 ** 64          # blinding factor r is drawn from [1, 2^64)
OFFSETS = [-2e-3, -1e-3, -2e-4, -1e-4, -5e-5, -1e-5, 0.0, 1e-5, 5e-5, 1e-4, 2e-4, 1e-3, 2e-3]


def crypto_direction(x, theta, pub, priv, user):
    """Exactly what the protocol does for one condition: encrypt both encodings,
    subtract homomorphically, blind with r in [1, 2^64), decrypt, take the sign."""
    cx = paillier.encrypt(encode(x), pub)
    ct = paillier.encrypt(encode(theta), pub)
    diff = paillier.subtract_encrypted(cx, ct, pub)
    blinded = paillier.scalar_multiply(diff, random.randrange(1, R_MAX), pub)
    return user._recover_direction(blinded)


def plain_dir(x, theta):
    return "left" if x <= theta else "right"


def enc_dir(x, theta):
    return "left" if encode(x) <= encode(theta) else "right"


def part1_audit():
    n = 3_000_000
    k = np.arange(n)
    trunc = np.trunc((k / 1e4) * 1e4).astype(np.int64)
    bad = int((trunc != k).sum())
    out = {"grid_values_tested": n, "grid_values_encoded_one_unit_low": bad,
           "fraction": bad / n,
           "examples": [(float(k[i] / 1e4), int(trunc[i])) for i in np.nonzero(trunc != k)[0][:5]],
           "encode_uses": "int(v * 10000) + 10**9 (truncation toward zero); "
                          "for v >= 0 this equals floor, for v < 0 it does not",
           "datasets": {}}
    for name in DATASET_NAMES:
        X, _ = load_dataset(name, data_dir="data")
        scaled = X * 1e4
        near = np.abs(scaled - np.round(scaled)) < 1e-6
        collapsed = 0
        for j in range(X.shape[1]):
            collapsed += len(np.unique(X[:, j])) - len(np.unique(np.trunc(scaled[:, j])))
        out["datasets"][name] = {
            "values": int(X.size),
            "negative_values": int((X < 0).sum()),
            "values_with_more_than_4_decimals": int((~near).sum()),
            "on_grid_values_encoded_one_unit_low": int(((np.trunc(scaled) != np.round(scaled)) & near).sum()),
            "distinct_raw_values_merged_by_encoding": int(collapsed),
        }
    return out


def part2_threshold_adjacent(pub, priv, depth):
    user = UserParty(pub, priv)
    out = {}
    for name in DATASET_NAMES:
        X, y = load_dataset(name, data_dir="data")
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
        t = clf.tree_
        thetas = sorted({float(t.threshold[i]) for i in range(t.node_count)
                         if t.children_left[i] != -1})
        rows = {}
        for off in OFFSETS:
            n = pl_vs_enc = enc_vs_cr = 0
            for th in thetas:
                x = th + off
                p, e, c = plain_dir(x, th), enc_dir(x, th), crypto_direction(x, th, pub, priv, user)
                n += 1
                pl_vs_enc += int(p != e)
                enc_vs_cr += int(e != c)
            rows[f"{off:+.0e}"] = {"cases": n, "plaintext_vs_encoded_differ": pl_vs_enc,
                                    "encoded_vs_protocol_differ": enc_vs_cr}
        out[name] = {"thresholds": len(thetas), "by_offset": rows}
        tot_enc_cr = sum(r["encoded_vs_protocol_differ"] for r in rows.values())
        print(f"[edge] {name}: {len(thetas)} thresholds; protocol differs from the "
              f"integer comparison in {tot_enc_cr} cases", flush=True)
    return out


def part3_extremes(pub, priv):
    user = UserParty(pub, priv)
    n = pub[0]
    cases = [-1e6, -1e5 - 1, -1e5, -1.0, -0.00005, 0.0, 126.5, 1e6, 1e12, 1e30, 1e200]
    out = []
    for x in cases:
        for theta in (0.0, 126.5):
            row = {"x": x, "theta": theta, "plaintext": plain_dir(x, theta)}
            try:
                enc_x = encode(x)
                row["encoded_x"] = enc_x if abs(enc_x) < 10 ** 30 else f"~1e{len(str(abs(enc_x))) - 1}"
                row["protocol"] = crypto_direction(x, theta, pub, priv, user)
                row["agrees"] = row["protocol"] == row["plaintext"]
            except Exception as ex:  # AssertionError from encrypt() for out-of-range plaintexts
                row["protocol"] = f"ERROR {type(ex).__name__}"
                row["agrees"] = False
            out.append(row)
    lo = -OFFSET / SCALE_FACTOR
    return {"supported_input_range_note":
            f"encode() is non-negative only for v >= {lo}; encrypt() rejects plaintexts outside [0, n)",
            "cases": out}


def part4_wraparound():
    bounds = {}
    for bits in (1024, 2048, 3072):
        n = 2 ** bits  # n is within a factor 2 of 2^bits; fine for an order of magnitude
        safe = (n // 2) // (R_MAX - 1)
        bounds[str(bits)] = {"safe_abs_encoded_difference_log10": round(math.log10(safe), 1),
                              "safe_abs_feature_difference_log10": round(math.log10(safe) - math.log10(SCALE_FACTOR), 1)}
    pub, priv = paillier.keygen(bits=128)
    user = UserParty(pub, priv)
    n = pub[0]
    safe = (n // 2) // (R_MAX - 1)
    demo = {"demo_key_bits": n.bit_length(), "safe_bound_encoded": safe, "trials": 200}
    for label, d in (("below_bound", safe // 4), ("far_above_bound", 2 ** 70)):
        wrong = 0
        for _ in range(200):
            c1 = paillier.encrypt(OFFSET + d, pub)
            c0 = paillier.encrypt(OFFSET, pub)
            diff = paillier.subtract_encrypted(c1, c0, pub)
            blinded = paillier.scalar_multiply(diff, random.randrange(1, R_MAX), pub)
            wrong += int(user._recover_direction(blinded) != "right")  # true answer: right
        demo[label] = {"difference": d, "wrong_sign": wrong}
    return {"condition": "sign is correct iff |x_enc - theta_enc| * r < n/2, with r in [1, 2^64)",
            "bounds_by_key_size": bounds, "demo": demo}


def encoded_predict(clf, X, scale):
    t = clf.tree_
    thr = [int(v * scale) for v in t.threshold]
    out = []
    for x in X:
        xe = [int(v * scale) for v in x]
        node = 0
        while t.children_left[node] != -1:
            node = t.children_left[node] if xe[t.feature[node]] <= thr[node] else t.children_right[node]
        out.append(int(clf.classes_[t.value[node].argmax()]))
    return np.array(out)


def plain64_predict(clf, X):
    t = clf.tree_
    out = []
    for x in X:
        node = 0
        while t.children_left[node] != -1:
            node = t.children_left[node] if x[t.feature[node]] <= t.threshold[node] else t.children_right[node]
        out.append(int(clf.classes_[t.value[node].argmax()]))
    return np.array(out)


def comparison_mismatch(clf, X, scale):
    """Every (record, internal node) pair: does the encoded comparison give the
    same branch as the float64 plaintext comparison x <= theta?"""
    t = clf.tree_
    bad = total = 0
    for node in range(t.node_count):
        if t.children_left[node] == -1:
            continue
        col = X[:, t.feature[node]]
        th = t.threshold[node]
        bad += int(((col <= th) != (np.trunc(col * scale) <= np.trunc(th * scale))).sum())
        total += len(col)
    return bad, total


def part5_precision_sweep(depth):
    """Three levels of mismatch per encoding scale:
       encoding        distinct raw feature values merged by the encoding
       comparison      branch decisions that differ from plaintext x <= theta
       classification  labels that differ from the plaintext tree
    (integer comparison; equal to the protocol whenever no wraparound occurs,
    which Part 2 checks against the real protocol)."""
    scales = [10 ** 2, 10 ** 4, 10 ** 6, 10 ** 8]
    out = {}
    for name in DATASET_NAMES:
        X, y = load_dataset(name, data_dir="data")
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
        ref = clf.predict(X)
        rec = {"records": len(X),
               "float64_plaintext_vs_sklearn_classification": int((plain64_predict(clf, X) != ref).sum()),
               "by_scale": {}}
        held = {s: [0, 0] for s in scales}
        splits = []
        for seed in range(5):
            Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            c2 = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xtr, ytr)
            splits.append((c2, Xte, c2.predict(Xte)))
        for s in scales:
            scaled = X * s
            tol = 1e-9 * np.maximum(1.0, np.abs(scaled))
            merged = sum(len(np.unique(X[:, j])) - len(np.unique(np.trunc(scaled[:, j])))
                         for j in range(X.shape[1]))
            beyond = int((np.abs(scaled - np.round(scaled)) > tol).sum())
            cb, ct = comparison_mismatch(clf, X, s)
            for c2, Xte, r2 in splits:
                held[s][0] += int((encoded_predict(c2, Xte, s) != r2).sum())
                held[s][1] += len(Xte)
            rec["by_scale"][f"1e{round(math.log10(s))}"] = {
                "encoding_distinct_values_merged": int(merged),
                "values_with_more_decimals_than_scale": beyond,
                "comparison_mismatch": {"mismatches": cb, "comparisons": ct},
                "classification_mismatch_all_records": int((encoded_predict(clf, X, s) != ref).sum()),
                "classification_mismatch_heldout_5_splits": {"mismatches": held[s][0], "records": held[s][1]},
            }
        out[name] = rec
        print(f"[edge] precision sweep {name}: classification mismatches (all records) "
              f"{ {k: v['classification_mismatch_all_records'] for k, v in rec['by_scale'].items()} }",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=5)
    a = ap.parse_args()
    pub, priv = paillier.keygen(bits=a.bits)

    res = {"meta": run_meta(bits=a.bits, depth=a.depth, scale=SCALE_FACTOR, offset=OFFSET,
                            split_seeds="0-4 (stratified 70/30)", tree_random_state=42)}
    res["part1_encoding_audit"] = part1_audit()
    print("[edge] part 1 done", flush=True)
    res["part2_threshold_adjacent"] = part2_threshold_adjacent(pub, priv, a.depth)
    res["part3_extreme_values"] = part3_extremes(pub, priv)
    print("[edge] part 3 done", flush=True)
    res["part4_wraparound"] = part4_wraparound()
    print("[edge] part 4:", json.dumps(res["part4_wraparound"]["demo"]), flush=True)
    res["part5_precision_sweep"] = part5_precision_sweep(a.depth)

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "exp_edge_cases.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[edge] wrote {path}")


if __name__ == "__main__":
    main()
