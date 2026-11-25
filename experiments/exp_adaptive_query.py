"""
exp_adaptive_query.py - how much of the model a querying user can recover.

Camera-ready comment addressed: R3-8 (restricted threat model; repeated
adaptive queries may reveal thresholds). The submitted paper states this
channel but does not quantify it. This script does, against the REAL protocol
(real Paillier, real blinding by r in [1, 2^64)), from the user's side.

Attack A - sign-only bisection. The user varies ONE feature and keeps only the
  sign it recovers for a rule, bisecting until the interval around the
  threshold is one encoding unit wide (1e-4). Reports queries needed per
  threshold; a plaintext tree needs the same number.
Attack B - gcd of repeated blinded values. The user sends the SAME input twice
  (or k times). The server draws a fresh blinding factor each time, so the two
  decrypted values are r1*d and r2*d with d = x_enc - theta_enc, and
  gcd(r1*d, r2*d) = d * gcd(r1, r2). With r bounded by 2^64 the gcd is d
  itself most of the time. Reports how many repeats recover theta exactly.
  This is the concrete reading of the sentence in Section III-D that the
  blinding "should be read as protecting the difference in transit and at the
  server, rather than as a guarantee against the querying party".

    python -m experiments.exp_adaptive_query
    python -m experiments.exp_adaptive_query --bits 512 --dataset pima

Output: results/exp_adaptive_query.json
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from system.rule_store import encode, OFFSET
from system.inference_engine import UserParty
from experiments.datasets import load_dataset
from experiments._meta import run_meta

R_MAX = 2 ** 64


def blinded_plain_value(x_enc, th_enc, pub, priv):
    """What the user obtains by decrypting one blinded difference (centered)."""
    n = pub[0]
    cx = paillier.encrypt(x_enc, pub)
    ct = paillier.encrypt(th_enc, pub)
    diff = paillier.subtract_encrypted(cx, ct, pub)
    blinded = paillier.scalar_multiply(diff, random.randrange(1, R_MAX), pub)
    m = paillier.decrypt(blinded, pub, priv)
    return m if m <= n // 2 else m - n


def attack_bisection(th_enc, lo_enc, hi_enc, pub, priv):
    """Sign-only bisection on the encoded integer grid. Returns (recovered, queries)."""
    queries = 0
    lo, hi = lo_enc, hi_enc          # invariant: theta_enc in [lo, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        queries += 1
        centered = blinded_plain_value(mid, th_enc, pub, priv)
        if centered <= 0:            # "left": mid <= theta
            lo = mid
        else:
            hi = mid
    return lo, queries


def attack_gcd(th_enc, x_enc, pub, priv, max_repeats=8):
    """Repeat one input; stop when the running gcd is unchanged twice in a row."""
    g, history = 0, []
    stable = 0
    for k in range(1, max_repeats + 1):
        g_new = math.gcd(g, abs(blinded_plain_value(x_enc, th_enc, pub, priv)))
        stable = stable + 1 if g_new == g else 0
        g = g_new
        history.append(g)
        if stable >= 2:
            return g, k
    return g, max_repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--dataset", default="pima")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max-thresholds", type=int, default=12)
    a = ap.parse_args()

    random.seed(0)
    pub, priv = paillier.keygen(bits=a.bits)
    X, y = load_dataset(a.dataset, data_dir="data")
    clf = DecisionTreeClassifier(max_depth=a.depth, random_state=42).fit(X, y)
    t = clf.tree_
    nodes = [i for i in range(t.node_count) if t.children_left[i] != -1][: a.max_thresholds]

    rows, ok_bis, ok_gcd = [], 0, 0
    for i in nodes:
        f, th = int(t.feature[i]), float(t.threshold[i])
        th_enc = encode(th)
        lo = encode(float(X[:, f].min())) - 1
        hi = encode(float(X[:, f].max())) + 1
        rec, q = attack_bisection(th_enc, lo, hi, pub, priv)
        x_enc = encode(float(X[:, f].min()))           # an input the attacker chooses
        g, reps = attack_gcd(th_enc, x_enc, pub, priv)
        gcd_theta = x_enc - g if x_enc >= th_enc else x_enc + g   # candidate; resolve sign by one sign query
        sign_x_above = blinded_plain_value(x_enc, th_enc, pub, priv) > 0
        gcd_theta = x_enc - g if sign_x_above else x_enc + g
        ok_bis += int(rec == th_enc)
        ok_gcd += int(gcd_theta == th_enc)
        rows.append({"feature": f, "threshold": th, "bisection_queries": q,
                     "bisection_recovers_encoded_threshold": rec == th_enc,
                     "gcd_repeats_until_stable": reps,
                     "gcd_recovers_encoded_threshold": gcd_theta == th_enc})
        print(f"[adaptive] feature {f:2d} theta={th:10.4f}: bisection {q} queries "
              f"({'ok' if rec == th_enc else 'miss'}); gcd {reps} repeats "
              f"({'ok' if gcd_theta == th_enc else 'miss'})", flush=True)

    res = {"meta": run_meta(bits=a.bits, dataset=a.dataset, depth=a.depth, rng_seed=0),
           "thresholds_attacked": len(rows),
           "bisection_success": ok_bis, "gcd_success": ok_gcd,
           "mean_bisection_queries": float(np.mean([r["bisection_queries"] for r in rows])),
           "mean_gcd_repeats": float(np.mean([r["gcd_repeats_until_stable"] for r in rows])),
           "rows": rows,
           "reading": "Thresholds are protected from the cloud, not from the querying user."}
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "exp_adaptive_query.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[adaptive] bisection {ok_bis}/{len(rows)} recovered, mean {res['mean_bisection_queries']:.1f} queries; "
          f"gcd {ok_gcd}/{len(rows)} recovered, mean {res['mean_gcd_repeats']:.1f} repeats")


if __name__ == "__main__":
    main()
