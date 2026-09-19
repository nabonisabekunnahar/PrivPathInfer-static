"""
exp_subtree_workload.py — User-side decryption time under
baseline_classify (no partitioning) vs subtree_classify at router
depths 1-4, for each dataset, at Paillier key size 1024. Averaged over
at least 10 queries per configuration, mean and standard deviation.

Reports only the within-scheme reduction percentage and the
underlying rule/decryption counts. Never a comparison against
plaintext or another scheme's timing — that question is out of scope
for this repository.

The partitioned store is built to be indistinguishable from the
baseline (c=1) store from the Cloud's side: same PRP feature-tag
scheme, same total rule count (padded with never-decrypted fresh
encryptions of real thresholds), same per-rule shape, no positional
tell (see system/subtree_partitioner.py). The savings measured here
are strictly on the User's decryption side.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import build_rule_store, RuleStore
from system.inference_engine import CloudParty, UserParty
from system.subtree_partitioner import (
    build_router_map, build_partitioned_rules, baseline_classify, subtree_classify,
)
from experiments.datasets import load_dataset, DATASET_NAMES

KEY_BITS = 1024
MAX_DEPTH = 5
RANDOM_STATE = 42
NUM_QUERIES = 10
ROUTER_DEPTHS = [1, 2, 3, 4]


def run_dataset(name):
    X, y = load_dataset(name, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    paths = extract_paths(clf)

    pub, priv = paillier.keygen(bits=KEY_BITS)

    baseline_store, prp_key = build_rule_store(paths, pub, c=1)
    user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=baseline_store.conceal_feature)
    cloud_baseline = CloudParty(pub, baseline_store)

    rng = np.random.RandomState(0)
    query_idxs = rng.choice(len(X), size=NUM_QUERIES, replace=False)

    # Baseline: precompute the cloud round (untimed), then time only
    # the decrypt+compare loop — isolating User-side decryption work.
    baseline_times, baseline_decrypts = [], []
    for idx in query_idxs:
        row = X[idx].tolist()
        tagged = user.tag_feature_vector(row)
        blinded = cloud_baseline.evaluate_round(tagged)

        user.decrypt_count = 0
        t0 = time.perf_counter()
        baseline_classify(user, baseline_store.rules, blinded, paths)
        baseline_times.append(time.perf_counter() - t0)
        baseline_decrypts.append(user.decrypt_count)

    baseline_mean = float(np.mean(baseline_times))
    baseline_std = float(np.std(baseline_times))

    router_results = {}
    for router_depth in ROUTER_DEPTHS:
        router_map = build_router_map(paths, router_depth)
        sub_rules, router_positions, subtree_positions = build_partitioned_rules(paths, router_depth, pub, prp_key)
        cloud_sub = CloudParty(pub, RuleStore(rules=sub_rules, public_key=pub, conceal_feature=True))

        assert len(sub_rules) == len(baseline_store.rules), (
            "partitioned store must match the baseline's rule count so the "
            "Cloud cannot infer that partitioning happened"
        )

        sub_times, sub_decrypts = [], []
        for idx in query_idxs:
            row = X[idx].tolist()
            tagged = user.tag_feature_vector(row)
            blinded = cloud_sub.evaluate_round(tagged)

            user.decrypt_count = 0
            t0 = time.perf_counter()
            subtree_classify(user, sub_rules, blinded, paths, router_map, router_positions, subtree_positions)
            sub_times.append(time.perf_counter() - t0)
            sub_decrypts.append(user.decrypt_count)

        sub_mean = float(np.mean(sub_times))
        sub_std = float(np.std(sub_times))
        reduction_pct = 100.0 * (baseline_mean - sub_mean) / baseline_mean

        num_subtree_positions = sum(len(v) for v in subtree_positions.values())
        num_padding_rules = len(sub_rules) - len(router_positions) - num_subtree_positions

        router_results[str(router_depth)] = {
            "decrypt_time_mean_sec": sub_mean,
            "decrypt_time_std_sec": sub_std,
            "reduction_pct": reduction_pct,
            "num_router_rules": len(router_positions),
            "num_subtree_rules_total": num_subtree_positions,
            "num_padding_rules": num_padding_rules,
            "num_total_rules": len(sub_rules),
            "mean_decrypt_count": float(np.mean(sub_decrypts)),
            "std_decrypt_count": float(np.std(sub_decrypts)),
        }
        print(f"[{name}] router_depth={router_depth}: reduction={reduction_pct:.2f}% "
              f"(mean decrypts {router_results[str(router_depth)]['mean_decrypt_count']:.1f} "
              f"vs baseline {np.mean(baseline_decrypts):.1f}, "
              f"padding={num_padding_rules})")

    return {
        "num_baseline_rules": len(baseline_store.rules),
        "baseline_decrypt_time_mean_sec": baseline_mean,
        "baseline_decrypt_time_std_sec": baseline_std,
        "mean_baseline_decrypt_count": float(np.mean(baseline_decrypts)),
        "std_baseline_decrypt_count": float(np.std(baseline_decrypts)),
        "router_depths": router_results,
    }


def main():
    results = {}
    for name in DATASET_NAMES:
        print(f"[exp_subtree_workload] running {name} ...")
        results[name] = run_dataset(name)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "exp_subtree_workload.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_subtree_workload] wrote {out_path}")


if __name__ == "__main__":
    main()
