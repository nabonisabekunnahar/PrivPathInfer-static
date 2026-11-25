"""
exp_efficiency.py - runtime, communication and storage against tree depth and
the deduplication parameter c, at one key size.

Camera-ready comments addressed: R3-5 (efficiency across tree depths and
ciphertext-sharing settings), R3-14 (overhead next to storage in the figures).
exp_keysize.py covers key sizes and the four datasets at depth 5; this script
covers depth and c.

    python -m experiments.exp_efficiency                     # PIMA, 1024-bit
    python -m experiments.exp_efficiency --datasets pima,heart --queries 10

For every dataset, depth in --depths (default 2,4,6,8,10,12) and c in
{1, 2, 4, 8, 16, 32, "max"}: rules, distinct ciphertexts, linkability, store
bytes, one-off store build time, and per-query user-encrypt / server /
user-decrypt / path-matching time, query and response bytes.
A tree may stop growing before the requested depth; the depth actually
reached is recorded as "depth_reached".

Expected: per-query time depends on the number of rules, not on c (every
rule is still evaluated); c changes storage, build time and linkability.

Output: results/exp_efficiency_<bits>.json
"""

import argparse
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import (
    build_rule_store, measure_storage_bytes, distinct_ciphertext_count, linkability,
)
from system.inference_engine import CloudParty, UserParty
from experiments.datasets import load_dataset
from experiments.exp_keysize import one_query, peak_rss_mb
from experiments._meta import run_meta

C_VALUES = [1, 2, 4, 8, 16, 32, "max"]


def mean(xs):
    return st.mean(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--depths", default="2,4,6,8,10,12")
    ap.add_argument("--datasets", default="pima")
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]

    pub, priv = paillier.keygen(bits=a.bits)
    ct_bytes = (2 * a.bits + 7) // 8
    res = {"meta": run_meta(bits=a.bits, queries_per_config=a.queries, depths=depths,
                            query_selection_seed=0, tree_random_state=42)}

    for name in a.datasets.split(","):
        X, y = load_dataset(name, data_dir="data")
        idx = np.random.RandomState(0).choice(len(X), size=min(a.queries, len(X)), replace=False)
        Xq = X[idx]
        res[name] = {}
        for depth in depths:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
            paths = extract_paths(clf)
            plain = clf.predict(Xq)
            for c in C_VALUES:
                t0 = time.perf_counter()
                store, prp_key = build_rule_store(paths, pub, c=c)
                build = time.perf_counter() - t0
                cloud = CloudParty(pub, store)
                user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=store.conceal_feature)
                one_query(user, cloud, Xq[0].tolist(), paths)  # warm-up
                te, ts, td, tm, agree = [], [], [], [], 0
                for i in range(len(Xq)):
                    lab, a1, b1, d1, m1, nq, nr = one_query(user, cloud, Xq[i].tolist(), paths)
                    te.append(a1); ts.append(b1); td.append(d1); tm.append(m1)
                    agree += int(lab == int(plain[i]))
                key = f"depth={depth},c={c}"
                res[name][key] = {
                    "depth_reached": int(clf.get_depth()),
                    "rules": len(store.rules),
                    "distinct_ciphertexts": distinct_ciphertext_count(store),
                    "linkability": linkability(store),
                    "store_bytes": measure_storage_bytes(store),
                    "build_sec": build,
                    "user_encrypt_sec": mean(te), "server_sec": mean(ts),
                    "user_decrypt_all_sec": mean(td), "path_matching_sec": mean(tm),
                    "end_to_end_sec": mean([w + x_ + y_ + z for w, x_, y_, z in zip(te, ts, td, tm)]),
                    "query_bytes": nq * ct_bytes, "response_bytes": nr * ct_bytes,
                    "agreement_with_plaintext": f"{agree}/{len(Xq)}",
                }
                print(f"[eff] {name} {key}: rules={len(store.rules)} "
                      f"build={build:.2f}s e2e={res[name][key]['end_to_end_sec']*1e3:.0f}ms "
                      f"store={res[name][key]['store_bytes']/1024:.1f}KB", flush=True)

    res["meta"]["peak_rss_mb"] = peak_rss_mb()
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"exp_efficiency_{a.bits}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eff] wrote {path}")


if __name__ == "__main__":
    main()
