"""
exp_baseline_fallback.py - does the table baseline's coverage failure come from
the particular construction (only observed cells are stored), or from
discretization itself? Two alternative coverage policies are compared with
the original "no entry -> no answer" policy.

Camera-ready comment addressed: R3-3 (fairness of the discretization baseline).

Policies for a held-out query whose discretized cell is NOT in the table:
  none      no answer                      (the submitted baseline)
  nearest   label of the nearest stored cell, L1 distance in bin-index space
  majority  the most frequent label in the table

Plaintext only (no cryptography): the encrypted table is a keyed bijection of
the cell keys, so coverage and labels are identical to the plaintext cells.
A sanity check against the real SDTC.classify_disc lookup is done on the first
200 held-out records of every run.

    python -m experiments.exp_baseline_fallback

For every dataset and bin count in {3, 5, 10, 20, 50}, over five stratified
70/30 splits (seeds 0-4), reports means of:
  no_entry_pct              held-out queries with no table entry
  answered_disagree_pct     disagreement with the plaintext tree among the
                            queries the table DOES answer
  nearest_disagree_pct      disagreement over ALL held-out queries when
                            unanswered ones fall back to the nearest cell
  majority_disagree_pct     the same with the majority-label fallback

The plaintext tree is fitted on the 70% training part (depth 5, seed 42); the
table is built from the same training part, as in the submitted experiment.
The original split seeds of the submitted Table II are not in the repository,
so no_entry_pct should be close to, not identical with, that table.

Output: results/exp_baseline_fallback.json
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from baseline.sdtc import SDTC
from experiments.datasets import load_dataset, DATASET_NAMES
from experiments._meta import run_meta

BIN_COUNTS = [3, 5, 10, 20, 50]
SEEDS = range(5)
DEPTH = 5


class TableBaseline(SDTC):
    """SDTC plus a plaintext copy of its cells, for evaluating coverage
    policies only. Construction is inherited unchanged."""

    def __init__(self, n_bins):
        super().__init__(n_bins=n_bins)
        self._cells, self._labels = [], []

    def _add_entry(self, disc_features, label):
        super()._add_entry(disc_features, label)
        self._cells.append(np.array(disc_features, dtype=int))
        self._labels.append(int(label))

    def finalize(self):
        self.C = np.vstack(self._cells)
        self.L = np.array(self._labels)
        self.cellset = {tuple(r) for r in self.C}
        self.cell_label = {tuple(r): l for r, l in zip(self.C, self.L)}
        self.majority = Counter(self._labels).most_common(1)[0][0]

    def nearest_label(self, disc):
        return int(self.L[np.abs(self.C - disc).sum(axis=1).argmin()])


def run(name, n_bins, seed):
    X, y = load_dataset(name, data_dir="data")
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = DecisionTreeClassifier(max_depth=DEPTH, random_state=42).fit(Xtr, ytr)
    plain = clf.predict(Xte)

    tb = TableBaseline(n_bins)
    tb.fit_encrypt(clf, Xtr)
    tb.finalize()
    D = tb.discretizer.transform(Xte)

    n = len(Xte)
    answered = wrong_answered = wrong_nearest = wrong_major = 0
    for i in range(n):
        key = tuple(D[i])
        if key in tb.cellset:
            answered += 1
            lab = tb.cell_label[key]
            bad = int(lab != int(plain[i]))
            wrong_answered += bad
            wrong_nearest += bad
            wrong_major += bad
        else:
            wrong_nearest += int(tb.nearest_label(D[i]) != int(plain[i]))
            wrong_major += int(tb.majority != int(plain[i]))
    # sanity: the encrypted lookup agrees with the plaintext cell set
    for i in range(min(200, n)):
        assert (tb.classify_disc(D[i]) is not None) == (tuple(D[i]) in tb.cellset)
    return {
        "no_entry_pct": 100.0 * (n - answered) / n,
        "answered_disagree_pct": (100.0 * wrong_answered / answered) if answered else None,
        "nearest_disagree_pct": 100.0 * wrong_nearest / n,
        "majority_disagree_pct": 100.0 * wrong_major / n,
        "table_entries": len(tb.cellset),
        "table_storage_bytes (modelled)": tb.get_storage_bytes(),
        "queries": n,
        "table_hits": answered,
        "table_misses": n - answered,
        "coverage_pct": 100.0 * answered / n,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    out = {"meta": run_meta(depth=DEPTH, splits=len(list(SEEDS)), split_seeds=list(SEEDS),
                            test_size=0.3, stratified=True, bin_counts=BIN_COUNTS,
                            discretization="equal-width bins per feature, fitted on the training part",
                            note="percentages and storage are means over splits; queries/hits/misses are sums")}
    for name in DATASET_NAMES:
        out[name] = {}
        for b in BIN_COUNTS:
            rs = [run(name, b, s) for s in SEEDS]
            agg = {k: mean([r[k] for r in rs]) for k in rs[0]}
            for k in ("queries", "table_hits", "table_misses"):   # counts are summed over the splits
                agg[k] = sum(r[k] for r in rs)
            out[name][str(b)] = agg
            a = agg["answered_disagree_pct"]
            print(f"[fallback] {name:13s} bins={b:2d}  no_entry={agg['no_entry_pct']:5.1f}%  "
                  f"answered_wrong={'  n/a' if a is None else f'{a:4.1f}%'}  "
                  f"nearest_wrong={agg['nearest_disagree_pct']:4.1f}%  "
                  f"majority_wrong={agg['majority_disagree_pct']:4.1f}%", flush=True)
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "exp_baseline_fallback.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[fallback] wrote {path}")


if __name__ == "__main__":
    main()
