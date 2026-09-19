"""
exp_sdtc_baseline.py — SDTC (Liang et al. 2021) baseline comparison:
classification disagreement rate against the plaintext tree at 5-bin
discretization, across all four datasets. SDTC is not part of
PrivPathInfer; this script exists only to produce the comparison point
the paper's evaluation section needs for the accuracy-loss claim.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier

from baseline.sdtc import SDTC
from experiments.datasets import load_dataset, DATASET_NAMES

MAX_DEPTH = 5
RANDOM_STATE = 42
N_BINS = 5


def run_dataset(name):
    X, y = load_dataset(name, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    plaintext_preds = clf.predict(X)

    sdtc = SDTC(n_bins=N_BINS)
    sdtc.fit_encrypt(clf, X)

    total = len(X)
    disagreements = 0
    for i in range(total):
        pred = sdtc.classify(X[i])
        if pred != int(plaintext_preds[i]):
            disagreements += 1

    return {
        "total_records": total,
        "disagreements": disagreements,
        "disagreement_pct": 100.0 * disagreements / total,
        "n_bins": N_BINS,
        "num_table_entries": sdtc.get_storage_size(),
    }


def main():
    results = {}
    for name in DATASET_NAMES:
        print(f"[exp_sdtc_baseline] running {name} ...")
        results[name] = run_dataset(name)
        stats = results[name]
        print(
            f"[exp_sdtc_baseline] {name}: {stats['disagreement_pct']:.2f}% disagreement "
            f"({stats['disagreements']}/{stats['total_records']}), "
            f"{stats['num_table_entries']} table entries"
        )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "exp_sdtc_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_sdtc_baseline] wrote {out_path}")


if __name__ == "__main__":
    main()
