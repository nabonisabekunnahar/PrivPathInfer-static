"""
exp_heldout_protocol.py - agreement between the plaintext tree and the FULL
encrypted two-round protocol on HELD-OUT records.

Camera-ready comment addressed: R3-6 (the held-out check in the submitted
paper evaluated the fixed-point comparison only, not the whole protocol).

For each dataset and each of --splits stratified 70/30 splits (seeds 0..N-1):
  fit a depth-5 tree on the 70% part, build the encrypted rule store,
  classify every held-out record with UserParty.classify (real Paillier,
  real blinding, real decryption) and compare with the plaintext tree.

    python -m experiments.exp_heldout_protocol                 # full run
    python -m experiments.exp_heldout_protocol --max-per-split 150   # faster

Rough cost at 1024-bit with gmpy2: about 0.2-0.5 s per record, so the full
run (~7,900 held-out records over 5 splits) takes on the order of 1-1.5 hours.
Results are written after every (dataset, split), so a crash loses nothing.

Output: results/exp_heldout_protocol.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.path_extractor import extract_paths
from system.rule_store import build_rule_store
from system.inference_engine import CloudParty, UserParty
from experiments.datasets import load_dataset, DATASET_NAMES
from experiments._meta import run_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--c", default="1", help='deduplication parameter: an integer or "max"')
    ap.add_argument("--max-per-split", type=int, default=None,
                    help="subsample this many held-out records per split (fixed seed)")
    ap.add_argument("--datasets", default=",".join(DATASET_NAMES))
    a = ap.parse_args()
    c = a.c if a.c == "max" else int(a.c)

    pub, priv = paillier.keygen(bits=a.bits)
    out_path = os.path.join("results", "exp_heldout_protocol.json")
    os.makedirs("results", exist_ok=True)
    results = {"meta": run_meta(bits=a.bits, c=a.c, depth=a.depth, splits=a.splits,
                                split_seeds=list(range(a.splits)), test_size=0.3, stratified=True,
                                max_per_split=a.max_per_split, tree_random_state=42)}

    for name in a.datasets.split(","):
        X, y = load_dataset(name, data_dir="data")
        results[name] = {"splits": []}
        for seed in range(a.splits):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y)
            if a.max_per_split and len(Xte) > a.max_per_split:
                sel = np.random.RandomState(seed).choice(len(Xte), a.max_per_split, replace=False)
                Xte = Xte[sel]
            clf = DecisionTreeClassifier(max_depth=a.depth, random_state=42).fit(Xtr, ytr)
            paths = extract_paths(clf)
            store, prp_key = build_rule_store(paths, pub, c=c)
            cloud = CloudParty(pub, store)
            user = UserParty(pub, priv, prp_key=prp_key, conceal_feature=store.conceal_feature)
            plain = clf.predict(Xte)

            bad, t0 = [], time.perf_counter()
            for i in range(len(Xte)):
                if user.classify(cloud, Xte[i].tolist(), paths) != int(plain[i]):
                    bad.append(i)
            rec = {"seed": seed, "heldout_records": len(Xte), "rules": len(store.rules),
                   "disagreements": len(bad), "disagreeing_indices": bad[:20],
                   "sec": time.perf_counter() - t0}
            results[name]["splits"].append(rec)
            print(f"[heldout] {name} split {seed}: {len(Xte) - len(bad)}/{len(Xte)} agree "
                  f"({rec['sec']:.0f}s)", flush=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        tot = sum(s["heldout_records"] for s in results[name]["splits"])
        dis = sum(s["disagreements"] for s in results[name]["splits"])
        results[name]["total_heldout"] = tot
        results[name]["total_disagreements"] = dis
        print(f"[heldout] {name} TOTAL: {tot - dis}/{tot} agree, {dis} disagreements", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[heldout] wrote {out_path}")


if __name__ == "__main__":
    main()
