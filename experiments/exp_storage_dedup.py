"""
exp_storage_dedup.py — two sweeps on PIMA at Paillier key size 1024:

1. Depth sweep: tree depths 2..12, rule store storage size (KB) at
   c=1 (maximum independence) vs c="max" (maximum sharing per
   (feature, threshold) pair), plus a third series: the SDTC baseline
   (Liang et al. 2021) at 5-bin discretization, one decision-table
   entry per leaf, no cross-path sharing.
2. Deduplication sweep: fixed depth 8, c in {1,2,4,8,16,32,"max"},
   reporting distinct ciphertext count, storage size (KB), and
   linkability (max rules sharing one ciphertext).

Each configuration is built 10 times; storage figures are reported as
mean/std. Ciphertext counts and linkability are exact combinatorial
properties of the (paths, c) construction, not measurements, so they
need no error bars (they are identical on every repetition). The SDTC
series is likewise an exact byte count (entry count x fixed entry
size), not a measurement with run-to-run variance, so it is reported
as a single number.

Storage byte-size methodology: a Paillier ciphertext lives in
Z_{n^2}, so its serialized size is 2 x the modulus bit-length, in
bytes (256 bytes at 1024-bit n). Each rule additionally carries a
fixed-size bookkeeping payload: a feature tag (16 bytes when
PRP-concealed at c=1, 4 bytes as a plaintext index otherwise), a
1-byte direction, a 4-byte path id, and a 1-byte label. See
system/rule_store.py:measure_storage_bytes for the exact accounting.
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
from baseline.sdtc import SDTC
from experiments.datasets import load_dataset

KEY_BITS = 1024
RANDOM_STATE = 42
REPS = 10
DEDUP_DEPTH = 8
DEDUP_C_VALUES = [1, 2, 4, 8, 16, 32, "max"]
DEPTH_RANGE = range(2, 13)
SDTC_N_BINS = 5


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

        sdtc = SDTC(n_bins=SDTC_N_BINS)
        sdtc.fit_encrypt(clf, X)
        sdtc_kb = sdtc.get_storage_bytes() / 1024.0

        results[str(depth)] = {
            "num_paths": len(paths),
            "num_occurrences": sum(len(p.conditions) for p in paths),
            "storage_kb_c1_mean": float(np.mean(kb_c1)),
            "storage_kb_c1_std": float(np.std(kb_c1)),
            "storage_kb_max_mean": float(np.mean(kb_max)),
            "storage_kb_max_std": float(np.std(kb_max)),
            "sdtc_num_entries": sdtc.get_storage_size(),
            "storage_kb_sdtc": sdtc_kb,
        }
        print(
            f"[depth_sweep] depth={depth}: c=1 {results[str(depth)]['storage_kb_c1_mean']:.1f}KB, "
            f"c=max {results[str(depth)]['storage_kb_max_mean']:.1f}KB, "
            f"sdtc(5-bin) {sdtc_kb:.1f}KB ({sdtc.get_storage_size()} entries)"
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
