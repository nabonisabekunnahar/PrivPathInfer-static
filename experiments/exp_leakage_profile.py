"""
exp_leakage_profile.py - what the hosting server can observe in the rule store,
as a function of the deduplication parameter c, plus a small controlled
frequency-analysis attack on the PRP feature tags.

Camera-ready comments addressed: R3-1 (a complete, parameter-dependent leakage
characterization) and R3-7 (measures beyond maximum linkability, and a
controlled inference experiment).

    python -m experiments.exp_leakage_profile
    python -m experiments.exp_leakage_profile --bits 512 --trials 20

Part A: structural profile, no attack. For every dataset (depth-5 tree) and
        for PIMA at depth 8 (the setting of Table III), and for each
        c in {1, 2, 4, 8, 16, 32, "max"}, computed from the stored rules only:
          distinct_ciphertexts / max_linkability / mean_linkability
          linked_pair_fraction   share of rule pairs the server can tell carry
                                 the same (feature, threshold) value
          ref_entropy_bits       entropy of the ciphertext-reference
                                 distribution (max possible: log2(rules))
          feature_tag_classes    number of distinct feature tags
          same_feature_pair_fraction  share of rule pairs the server can tell
                                 test the same feature (true at c=1 through
                                 equal PRP tags, and at c>1 through the index)
          feature_identity       "PRP tag (equality only)" at c=1,
                                 "index in the clear" at c>1
Part B: rank-frequency attack on the c=1 PRP tags. The adversary sees how many
        rules carry each tag, trains its own tree on a disjoint half of the
        public data, and matches tags to features by frequency rank. Accuracy
        is the share of rules whose feature is guessed correctly, against a
        random-assignment chance level. (At c>1 the index is in the clear, so
        feature identity is known by construction.)

Output: results/exp_leakage_profile.json
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from crypto import paillier
from crypto.prf_prp import prp_int
from system.path_extractor import extract_paths
from dataclasses import fields
from system.rule_store import build_rule_store, EncryptedRule
from experiments.datasets import load_dataset, DATASET_NAMES
from experiments._meta import run_meta

C_VALUES = [1, 2, 4, 8, 16, 32, "max"]


def pairs(counts):
    return sum(k * (k - 1) // 2 for k in counts)


def profile(store, paths, prp_key):
    R = len(store.rules)
    ct = Counter(r.threshold_ct for r in store.rules)
    tg = Counter(r.feature_tag for r in store.rules)
    total_pairs = R * (R - 1) // 2
    k = list(ct.values())
    H = -sum((x / R) * math.log2(x / R) for x in k)
    used = sorted({cd.feature_index for p in paths for cd in p.conditions})
    expected = {prp_int(prp_key, f) for f in used} if store.conceal_feature else set(used)
    shared_rules = sum(x for x in k if x >= 2)
    return {
        "observable_rule_fields": [f.name for f in fields(EncryptedRule)],
        "rules": R,
        "paths": len({r.path_id for r in store.rules}),
        "conditions_per_path": sorted(Counter(r.path_id for r in store.rules).values()),
        "leaf_label_counts": dict(Counter({r.path_id: r.label for r in store.rules}.values())),
        "distinct_ciphertexts": len(ct),
        "max_linkability": max(k),
        "mean_linkability": R / len(ct),
        "reference_histogram (ciphertexts referenced k times)": dict(sorted(Counter(k).items())),
        "rules_sharing_a_ciphertext": shared_rules,
        "rules_sharing_a_ciphertext_fraction": shared_rules / R,
        "linked_pair_fraction": pairs(k) / total_pairs,
        "ref_entropy_bits": H,
        "ref_entropy_max_bits": math.log2(R),
        "feature_tag_classes": len(tg),
        "tag_frequency (rules per tag, descending)": sorted(tg.values(), reverse=True),
        "same_feature_pair_fraction": pairs(tg.values()) / total_pairs,
        "feature_identity": "PRP tag (equality only)" if store.conceal_feature else "index in the clear",
        "tag_is_deterministic_per_feature": set(tg) == expected and len(tg) == len(used),
    }


def part_a(pub):
    out = {}
    settings = [(n, 5) for n in DATASET_NAMES] + [("pima", 8)]
    for name, depth in settings:
        X, y = load_dataset(name, data_dir="data")
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
        paths = extract_paths(clf)
        rows = {}
        for c in C_VALUES:
            store, prp_key = build_rule_store(paths, pub, c=c)
            rows[str(c)] = profile(store, paths, prp_key)
        out[f"{name}_depth{depth}"] = rows
        r1, rm = rows["1"], rows["max"]
        print(f"[leak] {name} d{depth}: linked pairs {r1['linked_pair_fraction']:.3f} (c=1) -> "
              f"{rm['linked_pair_fraction']:.3f} (c=max); tag classes {r1['feature_tag_classes']}", flush=True)
    return out


def freq_attack(X, y, depth, seed, pub):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(X))
    half = len(X) // 2
    vi, ai = perm[:half], perm[half:]
    victim = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X[vi], y[vi])
    aux = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X[ai], y[ai])
    store, prp_key = build_rule_store(extract_paths(victim), pub, c=1)
    truth = {prp_int(prp_key, f): f for f in range(X.shape[1])}
    tag_counts = Counter(r.feature_tag for r in store.rules)
    aux_counts = Counter(cd.feature_index for p in extract_paths(aux) for cd in p.conditions)
    tags = [t for t, _ in tag_counts.most_common()]
    feats = [f for f, _ in aux_counts.most_common()]
    guess = {t: feats[i] for i, t in enumerate(tags) if i < len(feats)}
    R = len(store.rules)
    hit = sum(n for t, n in tag_counts.items() if guess.get(t) == truth[t]) / R
    chance = []
    for _ in range(200):
        sh = feats[:]
        random.shuffle(sh)
        g = {t: sh[i] for i, t in enumerate(tags) if i < len(sh)}
        chance.append(sum(n for t, n in tag_counts.items() if g.get(t) == truth[t]) / R)
    return hit, float(np.mean(chance))


def part_b(pub, trials, depth):
    out = {}
    for name in DATASET_NAMES:
        X, y = load_dataset(name, data_dir="data")
        res = [freq_attack(X, y, depth, s, pub) for s in range(trials)]
        hit = np.array([r[0] for r in res])
        ch = np.array([r[1] for r in res])
        out[name] = {"trials": trials, "attack_rule_accuracy_mean": float(hit.mean()),
                     "attack_rule_accuracy_std": float(hit.std()),
                     "random_assignment_mean": float(ch.mean())}
        print(f"[leak] attack {name}: {hit.mean():.2f} +/- {hit.std():.2f} "
              f"(chance {ch.mean():.2f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--depth", type=int, default=5)
    a = ap.parse_args()
    pub, _ = paillier.keygen(bits=a.bits)
    res = {"meta": run_meta(bits=a.bits, depth=a.depth, attack_trials=a.trials,
                            attack_seeds="0..trials-1 (victim/aux split of the data)"),
           "part_a_structural_profile": part_a(pub),
           "part_b_frequency_attack_c1": part_b(pub, a.trials, a.depth)}
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "exp_leakage_profile.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[leak] wrote {path}")


if __name__ == "__main__":
    main()
