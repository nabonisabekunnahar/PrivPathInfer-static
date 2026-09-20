"""
exp_bin_sweep.py — sweep the SDTC baseline's discretization bin count
(3, 5, 10, 20, 50) across all four datasets, holding the tree and
encoding fixed (same max_depth=5, random_state=42, equal-width
strategy as exp_sdtc_baseline.py's single 5-bin figure).

Tests whether the disagreement reported for the 5-bin SDTC baseline is
an artifact of that specific bin count, or a persistent property of
discretization itself: for each (dataset, n_bins), records the
disagreement rate against the plaintext classifier, the number of
distinct decision-table entries, and the table's serialized size in
KB (SDTC.get_storage_bytes(), the same accounting
exp_storage_dedup.py's SDTC series uses). Finer binning is expected to
reduce disagreement while growing the table — recording both turns a
single number into a trade-off curve.

Touches only baseline/ — the tree, the encoding, and everything in
system/ are unchanged.
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
BIN_COUNTS = [3, 5, 10, 20, 50]


def run_dataset(name):
    X, y = load_dataset(name, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    plaintext_preds = clf.predict(X)
    total = len(X)

    per_bin = {}
    for n_bins in BIN_COUNTS:
        sdtc = SDTC(n_bins=n_bins)
        sdtc.fit_encrypt(clf, X)

        disagreements = 0
        for i in range(total):
            pred = sdtc.classify(X[i])
            if pred != int(plaintext_preds[i]):
                disagreements += 1

        per_bin[str(n_bins)] = {
            "disagreements": disagreements,
            "disagreement_pct": 100.0 * disagreements / total,
            "num_table_entries": sdtc.get_storage_size(),
            "storage_kb": sdtc.get_storage_bytes() / 1024.0,
        }
        print(
            f"[{name}] n_bins={n_bins}: {per_bin[str(n_bins)]['disagreement_pct']:.2f}% disagreement "
            f"({disagreements}/{total}), {sdtc.get_storage_size()} entries, "
            f"{per_bin[str(n_bins)]['storage_kb']:.2f}KB"
        )

    return {"total_records": total, "bin_counts": per_bin}


def main():
    results = {}
    for name in DATASET_NAMES:
        print(f"[exp_bin_sweep] running {name} ...")
        results[name] = run_dataset(name)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "exp_bin_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_bin_sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
