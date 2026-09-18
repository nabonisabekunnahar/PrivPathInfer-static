"""
exp_fidelity.py — classification agreement between the plaintext tree
and the full secure two-round protocol, at Paillier key size 1024 and
deduplication parameter c=1, for every record of every dataset.

This experiment runs a single full pass per dataset, not ten. Every
other experiment in this repository reports mean/std over >= 10
repetitions because the quantity being measured (storage bytes,
decryption wall-clock time) has real run-to-run variance. Agreement
here has none: sign recovery is exact whenever |r * (feature -
threshold)| < n/2, which holds by an astronomical margin at a 1024-bit
key (n/2 is a ~1024-bit number; r and the encoded difference together
are nowhere near that size). tests/test_all.py already exercises this
property directly. Repeating a ~55-minute full-dataset pass ten times
would spend hours reconfirming a result with zero variance, so we
report one full pass per dataset instead, matching the reporting
format specified for this experiment (total records, exact matches,
disagreements, percentage agreement).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import build_rule_store
from system.inference_engine import CloudParty, UserParty
from experiments.datasets import load_dataset, DATASET_NAMES

KEY_BITS = 1024
MAX_DEPTH = 5
RANDOM_STATE = 42


def run_fidelity(name, data_dir="data"):
    X, y = load_dataset(name, data_dir=data_dir)
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    clf.fit(X, y)
    paths = extract_paths(clf)

    pub, priv = paillier.keygen(bits=KEY_BITS)
    store, prp_key = build_rule_store(paths, pub, c=1)
    cloud = CloudParty(pub, store)
    user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=store.conceal_feature)

    plaintext_preds = clf.predict(X)

    total = len(X)
    disagreements = 0
    t0 = time.perf_counter()
    for i in range(total):
        secure_pred = user.classify(cloud, X[i].tolist(), paths)
        if secure_pred != int(plaintext_preds[i]):
            disagreements += 1
    elapsed = time.perf_counter() - t0

    matches = total - disagreements
    agreement_pct = 100.0 * matches / total

    return {
        "total_records": total,
        "exact_matches": matches,
        "disagreements": disagreements,
        "agreement_pct": agreement_pct,
        "num_paths": len(paths),
        "num_rules": len(store.rules),
        "wall_clock_sec": elapsed,
    }


def main():
    results = {}
    for name in DATASET_NAMES:
        print(f"[exp_fidelity] running {name} ...")
        stats = run_fidelity(name)
        results[name] = stats
        print(
            f"[exp_fidelity] {name}: {stats['exact_matches']}/{stats['total_records']} "
            f"matched ({stats['agreement_pct']:.4f}%), disagreements={stats['disagreements']}, "
            f"{stats['wall_clock_sec']:.1f}s"
        )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "exp_fidelity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_fidelity] wrote {out_path}")


if __name__ == "__main__":
    main()
