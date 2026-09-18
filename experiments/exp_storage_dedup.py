"""
exp_storage_dedup.py — two sweeps on PIMA at Paillier key size 1024:

1. Depth sweep: tree depths 2..12, rule store storage size (KB) at
   c=1 (maximum independence) vs c="max" (maximum sharing per
   (feature, threshold) pair).
2. Deduplication sweep: fixed depth 8, c in {1,2,4,8,16,32,"max"},
   reporting distinct ciphertext count, storage size (KB), and
   linkability (max rules sharing one ciphertext).

Each configuration is built 10 times; storage figures are reported as
mean/std. Ciphertext counts and linkability are exact combinatorial
properties of the (paths, c) construction, not measurements, so they
need no error bars (they are identical on every repetition).
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
from experiments.datasets import load_dataset

KEY_BITS = 1024
RANDOM_STATE = 42
REPS = 10
DEDUP_DEPTH = 8
DEDUP_C_VALUES = [1, 2, 4, 8, 16, 32, "max"]
DEPTH_RANGE = range(2, 13)


def _build_and_measure(paths, pub, c):
    store, _prp_key = build_rule_store(paths, pub, c)
    return {
        "ciphertext_count": distinct_ciphertext_count(store),
        "linkability": linkability(store),
        "storage_kb": measure_storage_bytes(store) / 1024.0,
    }


def run_depth_sweep(X, y):
    results = {}
    for depth in DEPTH_RANGE:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(X, y)
        paths = extract_paths(clf)

        kb_c1, kb_max = [], []
        for _rep in range(REPS):
            pub, _priv = paillier.keygen(bits=KEY_BITS)
            kb_c1.append(_build_and_measure(paths, pub, 1)["storage_kb"])
            kb_max.append(_build_and_measure(paths, pub, "max")["storage_kb"])

        results[str(depth)] = {
            "num_paths": len(paths),
            "num_occurrences": sum(len(p.conditions) for p in paths),
            "storage_kb_c1_mean": float(np.mean(kb_c1)),
            "storage_kb_c1_std": float(np.std(kb_c1)),
            "storage_kb_max_mean": float(np.mean(kb_max)),
            "storage_kb_max_std": float(np.std(kb_max)),
        }
        print(
            f"[depth_sweep] depth={depth}: c=1 {results[str(depth)]['storage_kb_c1_mean']:.1f}KB, "
            f"c=max {results[str(depth)]['storage_kb_max_mean']:.1f}KB"
        )
    return results


def run_dedup_ablation(X, y):
    clf = DecisionTreeClassifier(max_depth=DEDUP_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    paths = extract_paths(clf)

    results = {}
    for c in DEDUP_C_VALUES:
        kb_values = []
        ciphertext_counts, linkabilities = [], []
        for _rep in range(REPS):
            pub, _priv = paillier.keygen(bits=KEY_BITS)
            stats = _build_and_measure(paths, pub, c)
            kb_values.append(stats["storage_kb"])
            ciphertext_counts.append(stats["ciphertext_count"])
            linkabilities.append(stats["linkability"])

        assert len(set(ciphertext_counts)) == 1, "ciphertext count should be exact, not measured"
        assert len(set(linkabilities)) == 1, "linkability should be exact, not measured"

        results[str(c)] = {
            "ciphertext_count": ciphertext_counts[0],
            "linkability": linkabilities[0],
            "storage_kb_mean": float(np.mean(kb_values)),
            "storage_kb_std": float(np.std(kb_values)),
        }
        print(
            f"[dedup_ablation] c={c}: ciphertexts={ciphertext_counts[0]}, "
            f"storage={results[str(c)]['storage_kb_mean']:.1f}KB, "
            f"linkability={linkabilities[0]}"
        )
    return results


def main():
    X, y = load_dataset("pima", data_dir="data")

    depth_results = run_depth_sweep(X, y)
    dedup_results = run_dedup_ablation(X, y)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "exp_storage_depth_sweep.json"), "w") as f:
        json.dump(depth_results, f, indent=2)
    with open(os.path.join("results", "exp_dedup_ablation.json"), "w") as f:
        json.dump(dedup_results, f, indent=2)
    print("[exp_storage_dedup] wrote results/exp_storage_depth_sweep.json and results/exp_dedup_ablation.json")


if __name__ == "__main__":
    main()
