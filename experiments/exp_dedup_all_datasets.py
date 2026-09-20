"""
exp_dedup_all_datasets.py — the deduplication (c) sweep from
exp_storage_dedup.py's ablation, run on all four datasets (not just
PIMA) at a fixed target depth of 8, the same 1024-bit key, and the
same c in {1, 2, 4, 8, 16, 32, "max"}.

Tests whether the storage/linkability trade-off generalizes across
datasets of very different shape (30 features vs 8, 3,658 records vs
303) rather than being a PIMA-specific artifact.

max_depth=8 is a ceiling on tree growth, not a target: a dataset can
produce a shallower tree if its splits reach pure leaves first. Where
that happens, the actually-reached depth is used and recorded
alongside the numbers (`actual_depth`) rather than forcing a depth the
data does not support.

Does not touch the tree-building logic, the encoding, or anything in
system/ beyond calling the same public functions
exp_storage_dedup.py already uses.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import (
    build_rule_store, distinct_ciphertext_count, linkability, measure_storage_bytes,
)
from experiments.datasets import load_dataset, DATASET_NAMES

KEY_BITS = 1024
RANDOM_STATE = 42
REPS = 10
TARGET_DEPTH = 8
C_VALUES = [1, 2, 4, 8, 16, 32, "max"]


def run_dataset(name):
    X, y = load_dataset(name, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=TARGET_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    actual_depth = clf.get_depth()
    paths = extract_paths(clf)

    per_c = {}
    for c in C_VALUES:
        kb_values = []
        ciphertext_counts, linkabilities = [], []
        for _rep in range(REPS):
            pub, _priv = paillier.keygen(bits=KEY_BITS)
            store, _prp_key = build_rule_store(paths, pub, c)
            kb_values.append(measure_storage_bytes(store) / 1024.0)
            ciphertext_counts.append(distinct_ciphertext_count(store))
            linkabilities.append(linkability(store))

        assert len(set(ciphertext_counts)) == 1, "ciphertext count should be exact, not measured"
        assert len(set(linkabilities)) == 1, "linkability should be exact, not measured"

        per_c[str(c)] = {
            "ciphertext_count": ciphertext_counts[0],
            "linkability": linkabilities[0],
            "storage_kb_mean": float(np.mean(kb_values)),
            "storage_kb_std": float(np.std(kb_values)),
        }
        print(
            f"[{name}] c={c}: ciphertexts={ciphertext_counts[0]}, "
            f"storage={per_c[str(c)]['storage_kb_mean']:.1f}KB, "
            f"linkability={linkabilities[0]}"
        )

    return {
        "target_depth": TARGET_DEPTH,
        "actual_depth": actual_depth,
        "num_paths": len(paths),
        "num_occurrences": sum(len(p.conditions) for p in paths),
        "c_values": per_c,
    }


def main():
    results = {}
    for name in DATASET_NAMES:
        print(f"[exp_dedup_all_datasets] running {name} ...")
        results[name] = run_dataset(name)
        depth_note = "" if results[name]["actual_depth"] == TARGET_DEPTH else " (SHALLOWER THAN TARGET)"
        print(
            f"[exp_dedup_all_datasets] {name}: actual_depth={results[name]['actual_depth']} "
            f"(target {TARGET_DEPTH}){depth_note}"
        )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "exp_dedup_all_datasets.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_dedup_all_datasets] wrote {out_path}")


if __name__ == "__main__":
    main()
