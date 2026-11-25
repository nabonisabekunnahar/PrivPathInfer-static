"""
exp_keysize.py - Paillier key-size sweep: per-phase timing, message sizes,
store size and peak memory.

Camera-ready comments addressed: key size (R3-2), end-to-end efficiency
(R3-5), overhead next to storage in the tables (R3-14), hardware and
reproducibility details (R3-15).

Run ONE key size per process so that peak memory is per key size:

    python -m experiments.exp_keysize --bits 1024
    python -m experiments.exp_keysize --bits 2048
    python -m experiments.exp_keysize --bits 3072

Options: --queries N (default 20), --datasets pima,heart,... (default all),
         --depth D (default 5, as in the fidelity experiment).

Output: results/exp_keysize_<bits>.json

What is measured, per dataset and per deduplication setting c in {1, "max"}:
  keygen_sec        one-off key generation
  build_sec         encrypting the whole rule store (institution side, one-off)
  user_encrypt_sec  user encrypts its feature vector (per query)
  server_sec        server computes all blinded differences (per query)
  user_decrypt_sec  user decrypts ALL returned ciphertexts (per query)
  per_decrypt_sec   cost of one decryption
  path_follow_est   mean comparisons on a decision path x per_decrypt_sec
                    (DERIVED, not measured: the cost if the user follows the
                    path and decrypts one ciphertext per level)
  query_bytes / response_bytes  ciphertexts sent each way
  store_bytes       output of measure_storage_bytes()
"""

import argparse
import json
import os
import platform
import statistics as st
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import (
    build_rule_store, measure_storage_bytes, distinct_ciphertext_count, linkability,
)
from system.inference_engine import CloudParty, UserParty
from experiments.datasets import load_dataset, DATASET_NAMES
from experiments._meta import run_meta

RANDOM_STATE = 42


def peak_rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2 ** 20
        except Exception:
            return None


def ms(xs):
    return {"mean": st.mean(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0}


def one_query(user, cloud, x, paths):
    """Returns label and the time of each phase: user encrypt, server compute,
    user decrypt (all returned ciphertexts), path matching (tree traversal)."""
    t0 = time.perf_counter()
    tagged = user.tag_feature_vector(x)
    t1 = time.perf_counter()
    blinded = cloud.evaluate_round(tagged)
    t2 = time.perf_counter()
    directions = [user._recover_direction(ct) for ct in blinded]
    t3 = time.perf_counter()
    matches = defaultdict(list)
    for rule, d in zip(cloud.rule_store.rules, directions):
        matches[rule.path_id].append(d == rule.direction)
    label = None
    for p in paths:
        m = matches.get(p.path_id, [])
        if len(m) == len(p.conditions) and all(m):
            label = p.label
            break
    t4 = time.perf_counter()
    return label, t1 - t0, t2 - t1, t3 - t2, t4 - t3, len(tagged), len(blinded)


def run_dataset(name, bits, pub, priv, n_queries, depth):
    X, y = load_dataset(name, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE).fit(X, y)
    paths = extract_paths(clf)
    rs = np.random.RandomState(0)
    idx = rs.choice(len(X), size=min(n_queries, len(X)), replace=False)
    Xq = X[idx]
    plain = clf.predict(Xq)
    comparisons = np.asarray(clf.decision_path(Xq).sum(axis=1)).ravel() - 1
    ct_bytes = (2 * bits + 7) // 8

    out = {"n_features": X.shape[1], "n_paths": len(paths), "queries": len(idx),
           "mean_comparisons_per_path": float(comparisons.mean())}
    for c in (1, "max"):
        t0 = time.perf_counter()
        store, prp_key = build_rule_store(paths, pub, c=c)
        build_sec = time.perf_counter() - t0
        cloud = CloudParty(pub, store)
        user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=store.conceal_feature)
        one_query(user, cloud, Xq[0].tolist(), paths)  # warm-up, not recorded

        te, ts, td, tm, agree = [], [], [], [], 0
        for i in range(len(Xq)):
            label, a, b, d, m_, nq, nr = one_query(user, cloud, Xq[i].tolist(), paths)
            te.append(a); ts.append(b); td.append(d); tm.append(m_)
            agree += int(label == int(plain[i]))
        per_dec = [d / nr for d in td]
        out[f"c={c}"] = {
            "rules": len(store.rules),
            "distinct_ciphertexts": distinct_ciphertext_count(store),
            "linkability": linkability(store),
            "store_bytes": measure_storage_bytes(store),
            "build_sec": build_sec,
            "user_encrypt_sec": ms(te),
            "encrypt_per_value_sec": st.mean(te) / X.shape[1],
            "server_sec": ms(ts),
            "server_per_comparison_sec": st.mean(ts) / nr,
            "user_decrypt_all_sec": ms(td),
            "path_matching_sec (tree traversal)": ms(tm),
            "per_decrypt_sec": ms(per_dec),
            "path_follow_est_sec (derived)": float(comparisons.mean() * st.mean(per_dec)),
            "end_to_end_sec (measured, decrypt-all)": ms([a + b + d + m_ for a, b, d, m_ in zip(te, ts, td, tm)]),
            "ciphertext_bytes": ct_bytes,
            "messages": {"client_to_server": 1, "server_to_client": 1, "round_trips": 1},
            "query_bytes": nq * ct_bytes,
            "response_bytes": nr * ct_bytes,
            "total_communication_bytes": (nq + nr) * ct_bytes,
            "agreement_with_plaintext": f"{agree}/{len(Xq)}",
        }
        print(f"[{bits}] {name} c={c}: rules={len(store.rules)} "
              f"enc={st.mean(te)*1e3:.1f}ms srv={st.mean(ts)*1e3:.1f}ms "
              f"dec_all={st.mean(td)*1e3:.1f}ms agree={agree}/{len(Xq)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--datasets", default=",".join(DATASET_NAMES))
    a = ap.parse_args()

    t0 = time.perf_counter()
    pub, priv = paillier.keygen(bits=a.bits)
    keygen_sec = time.perf_counter() - t0
    print(f"[{a.bits}] keygen {keygen_sec:.1f}s  (gmpy2={paillier.GMPY2_AVAILABLE})", flush=True)

    res = {"meta": run_meta(bits=a.bits, keygen_sec=keygen_sec, gmpy2=paillier.GMPY2_AVAILABLE,
                            tree_depth=a.depth, queries_per_config=a.queries,
                            query_selection_seed=0, tree_random_state=RANDOM_STATE)}
    for name in a.datasets.split(","):
        res[name] = run_dataset(name, a.bits, pub, priv, a.queries, a.depth)
    res["meta"]["peak_rss_mb"] = peak_rss_mb()
    res["meta"]["total_sec"] = time.perf_counter() - t0

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"exp_keysize_{a.bits}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{a.bits}] peak RSS {res['meta']['peak_rss_mb']} MB; wrote {path}")


if __name__ == "__main__":
    main()
