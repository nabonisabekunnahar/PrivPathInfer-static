"""
subtree_partitioner.py — router/subtree split for reduced decryption
workload.

Each path's conditions are split at a router-depth parameter d into a
shared prefix (depth <= d — identical across every path that passes
through the same tree node, and deduplicated into a small set of
"router rules") and a path-specific suffix ("subtree rules", one per
path, no sharing).

The private mapping from each possible router outcome to the subtree
(the set of path_ids) it selects is built once and held only by the
User; it is never transmitted to the Cloud.

For the Cloud's view to genuinely match the unpartitioned baseline's
(the claim this optimization depends on), the partitioned rule list
must be indistinguishable from the baseline's in every property the
Cloud can observe:
    - same feature-tag scheme (PRP-concealed, not raw indices)
    - same total rule count (padded with fresh encryptions of real,
      never-decrypted thresholds so router-plus-subtree rules don't
      undershoot the baseline count)
    - same shape per rule (every rule carries a real, non-negative
      path_id/label — never a -1 sentinel that would mark it as a
      router row)
    - no positional tell (the final list is shuffled with a fixed
      seed; router_positions/subtree_positions are built against the
      shuffled order)

During inference the Cloud still returns blinded results for every
rule in the flat list — exactly as in the unpartitioned baseline
protocol, with no indication of which rules are routers, subtree
rules, or padding. The savings are entirely on the User's side:
baseline_classify decrypts every rule; subtree_classify decrypts only
the small deduplicated router set first, uses the private map to
identify which subtree applies, and decrypts only that subtree's
rules — skipping every other subtree (and all the padding) entirely,
exactly as it already skips other subtrees.
"""

import random
from collections import defaultdict

from crypto import paillier
from crypto.prf_prp import prp_int
from system.rule_store import EncryptedRule, encode

# Fixed seed for both the padding-condition draw and the final shuffle.
# router_positions/subtree_positions are only meaningful together with
# a rule list shuffled under this same seed.
PARTITION_SHUFFLE_SEED = 1337


def _prefix_suffix(path, router_depth):
    return path.conditions[:router_depth], path.conditions[router_depth:]


def build_router_map(paths, router_depth):
    """
    {router_key: [path_id, ...]}, where router_key is the tuple of
    (feature_index, threshold, direction) taken through the shared
    prefix. This mapping is private to the User and is never
    transmitted to the Cloud.
    """
    router_map = defaultdict(list)
    for path in paths:
        prefix, _ = _prefix_suffix(path, router_depth)
        router_key = tuple((c.feature_index, c.threshold, c.direction) for c in prefix)
        router_map[router_key].append(path.path_id)
    return dict(router_map)


def build_partitioned_rules(paths, router_depth, public_key, prp_key):
    """
    Build the single flat rule list the Cloud processes, made
    indistinguishable from the unpartitioned (c=1) baseline store for
    the same tree: same feature-tag scheme, same total rule count,
    same per-rule shape, no positional tell.

    Construction:
        1. Router rules — one per unique (feature_index, threshold)
           pair occurring in any path's shared prefix (depth <=
           router_depth), PRP-tagged with `prp_key` exactly as
           build_rule_store does at c=1. Since a router rule is
           shared by every path through that prefix, there is no
           single honest owner for its path_id/label; it is assigned
           the path_id/label of the first path (in declaration order)
           that uses it, so the rule is shaped like any other rather
           than carrying a sentinel. The User never reads these
           fields for router rules — it locates them via
           router_positions — so this changes nothing functionally.
        2. Subtree rules — one per path-specific suffix condition, no
           sharing, PRP-tagged the same way.
        3. Padding — fresh Paillier encryptions of real thresholds
           drawn from the tree (tagged with a real PRP'd feature
           index, a real direction, and a real path_id/label copied
           from the same condition), added until the total rule count
           equals what the unpartitioned baseline store would have
           for this tree (one rule per condition occurrence across
           all paths). These are never decrypted by the User, exactly
           like the subtree rules of a rejected candidate.
        4. The whole list is shuffled under a fixed seed so position
           alone reveals nothing; router_positions/subtree_positions
           are built against the post-shuffle indices.

    Args:
        prp_key: the same PRP key used to build the baseline (c=1)
            store, so the same UserParty can query both stores with
            matching tags.

    Returns:
        rules:              List[EncryptedRule], same length as the
            unpartitioned baseline store's rule list for this tree.
        router_positions:   {(feature_index, threshold): index into rules}
        subtree_positions:  {path_id: [index into rules, ...]}
    """
    router_rules = []        # [(key, EncryptedRule)]
    router_key_seen = set()

    for path in paths:
        prefix, _ = _prefix_suffix(path, router_depth)
        for cond in prefix:
            key = (cond.feature_index, cond.threshold)
            if key not in router_key_seen:
                router_key_seen.add(key)
                ct = paillier.encrypt(encode(cond.threshold), public_key)
                rule = EncryptedRule(
                    feature_tag=prp_int(prp_key, cond.feature_index),
                    threshold_ct=ct,
                    direction=cond.direction,
                    path_id=path.path_id,
                    label=path.label,
                )
                router_rules.append((key, rule))

    subtree_rules = []        # [(path_id, EncryptedRule)]
    for path in paths:
        _, suffix = _prefix_suffix(path, router_depth)
        for cond in suffix:
            ct = paillier.encrypt(encode(cond.threshold), public_key)
            rule = EncryptedRule(
                feature_tag=prp_int(prp_key, cond.feature_index),
                threshold_ct=ct,
                direction=cond.direction,
                path_id=path.path_id,
                label=path.label,
            )
            subtree_rules.append((path.path_id, rule))

    # Pad to the unpartitioned baseline's rule count: one rule per
    # condition occurrence across all paths.
    all_conditions = [(path, cond) for path in paths for cond in path.conditions]
    target_total = len(all_conditions)
    n_padding = max(0, target_total - (len(router_rules) + len(subtree_rules)))

    rng = random.Random(PARTITION_SHUFFLE_SEED)

    padding_rules = []
    for _ in range(n_padding):
        pad_path, pad_cond = rng.choice(all_conditions)
        ct = paillier.encrypt(encode(pad_cond.threshold), public_key)
        rule = EncryptedRule(
            feature_tag=prp_int(prp_key, pad_cond.feature_index),
            threshold_ct=ct,
            direction=pad_cond.direction,
            path_id=pad_path.path_id,
            label=pad_path.label,
        )
        padding_rules.append(rule)

    tagged = (
        [("router", key, rule) for key, rule in router_rules]
        + [("subtree", path_id, rule) for path_id, rule in subtree_rules]
        + [("pad", None, rule) for rule in padding_rules]
    )
    rng.shuffle(tagged)

    rules = []
    router_positions = {}
    subtree_positions = defaultdict(list)
    for kind, ident, rule in tagged:
        idx = len(rules)
        rules.append(rule)
        if kind == "router":
            router_positions[ident] = idx
        elif kind == "subtree":
            subtree_positions[ident].append(idx)

    return rules, router_positions, dict(subtree_positions)


def baseline_classify(user, cloud_rules, blinded, paths):
    """
    No partitioning: decrypts every rule. `blinded` is the Cloud's
    round result for `cloud_rules` (same order), computed beforehand
    so this function measures only the User's decryption workload.
    """
    matches_by_path = defaultdict(list)
    for rule, blinded_ct in zip(cloud_rules, blinded):
        recovered = user._recover_direction(blinded_ct)
        matches_by_path[rule.path_id].append(recovered == rule.direction)

    for path in paths:
        conditions_matched = matches_by_path.get(path.path_id, [])
        if len(conditions_matched) == len(path.conditions) and all(conditions_matched):
            return path.label
    return None


def subtree_classify(user, cloud_rules, blinded, paths, router_map, router_positions, subtree_positions):
    """
    Router-then-selective-decrypt: decrypts the small deduplicated
    router set first, uses the private router_map to identify which
    subtree(s) survive, and decrypts only those subtrees' rules.
    """
    recovered = {}
    for key, idx in router_positions.items():
        recovered[key] = user._recover_direction(blinded[idx])

    candidate_path_ids = set()
    for router_key, path_ids in router_map.items():
        if all(recovered.get((f, t)) == d for (f, t, d) in router_key):
            candidate_path_ids.update(path_ids)

    for path in paths:
        if path.path_id not in candidate_path_ids:
            continue
        ok = True
        for idx in subtree_positions.get(path.path_id, []):
            rule = cloud_rules[idx]
            if user._recover_direction(blinded[idx]) != rule.direction:
                ok = False
                break
        if ok:
            return path.label
    return None
