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

During inference the Cloud still returns blinded results for every
rule in the flat list — exactly as in the unpartitioned baseline
protocol, with no indication of which rules are routers. The savings
are entirely on the User's side: baseline_classify decrypts every
rule; subtree_classify decrypts only the small deduplicated router set
first, uses the private map to identify which subtree applies, and
decrypts only that subtree's rules — skipping every other subtree
entirely.
"""

from collections import defaultdict

from crypto import paillier
from system.rule_store import EncryptedRule, encode


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


def build_partitioned_rules(paths, router_depth, public_key):
    """
    Build the single flat rule list the Cloud processes: one
    deduplicated ciphertext per unique (feature_index, threshold) pair
    occurring in the shared prefix (the router rules), followed by one
    rule per path-specific suffix condition (the subtree rules, no
    sharing). The Cloud sees one undifferentiated list.

    Returns:
        rules:              List[EncryptedRule]
        router_positions:   {(feature_index, threshold): index into rules}
        subtree_positions:  {path_id: [index into rules, ...]}
    """
    rules = []
    router_positions = {}
    subtree_positions = defaultdict(list)

    for path in paths:
        prefix, _ = _prefix_suffix(path, router_depth)
        for cond in prefix:
            key = (cond.feature_index, cond.threshold)
            if key not in router_positions:
                ct = paillier.encrypt(encode(cond.threshold), public_key)
                rules.append(EncryptedRule(
                    feature_tag=cond.feature_index,
                    threshold_ct=ct,
                    direction=cond.direction,
                    path_id=-1,
                    label=-1,
                ))
                router_positions[key] = len(rules) - 1

    for path in paths:
        _, suffix = _prefix_suffix(path, router_depth)
        for cond in suffix:
            ct = paillier.encrypt(encode(cond.threshold), public_key)
            rules.append(EncryptedRule(
                feature_tag=cond.feature_index,
                threshold_ct=ct,
                direction=cond.direction,
                path_id=path.path_id,
                label=path.label,
            ))
            subtree_positions[path.path_id].append(len(rules) - 1)

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
