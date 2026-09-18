"""
inference_engine.py — the two-round secure comparison protocol.

Leakage:
    This protocol round adds NOTHING to the Cloud's view beyond
    semantically-secure Paillier ciphertexts: the User's tagged query
    ciphertexts, and the blinded difference ciphertexts the Cloud
    returns. The Cloud never decrypts anything — CloudParty has no
    private key and no decrypt method (enforced structurally below via
    __slots__, so no future code path can silently add cloud-side
    decryption).

    That said, the rule store the Cloud already holds keeps path_id,
    direction, and label in the clear (see system/rule_store.py). So
    the Cloud's TOTAL view — rule store plus protocol round — is the
    shape of the tree and its leaf labels. That is NOT the empty set.
    Only the narrower claim is true: this protocol round, by itself,
    reveals nothing beyond what the rule store already discloses.
"""

import random
from collections import defaultdict

from crypto import paillier
from crypto.prf_prp import prp_int
from system.rule_store import encode


class CloudParty:
    """
    Holds only the Paillier public key and the rule store. No private
    key, no decrypt method — __slots__ below means no code path can
    ever attach one without raising AttributeError.
    """

    __slots__ = ("public_key", "rule_store")

    def __init__(self, public_key, rule_store):
        self.public_key = public_key
        self.rule_store = rule_store

    def evaluate_round(self, tagged_feature_cts, rng=None):
        """
        For every rule in the store, compute the homomorphic difference
        Enc(feature) (.) Enc(threshold)^{-1} = Enc(feature - threshold)
        and blind it by a fresh random positive scalar r before
        returning it.

        Args:
            tagged_feature_cts: {feature_tag: ciphertext} for the
                User's query, one ciphertext per feature.
            rng: optional random.Random for the blinding scalars.

        Returns:
            list[int]: blinded ciphertexts, same order as
            self.rule_store.rules.
        """
        if rng is None:
            rng = random
        blinded = []
        for rule in self.rule_store.rules:
            feature_ct = tagged_feature_cts[rule.feature_tag]
            diff_ct = paillier.subtract_encrypted(feature_ct, rule.threshold_ct, self.public_key)
            r = rng.randrange(1, 2 ** 64)
            blinded.append(paillier.scalar_multiply(diff_ct, r, self.public_key))
        return blinded


class UserParty:
    """
    Holds the Paillier key pair (and, when the rule store conceals
    feature identity at c=1, the PRP key used to tag query features to
    match the store's feature_tag scheme).
    """

    def __init__(self, public_key, private_key, prp_key=None, conceal_feature=False):
        self.public_key = public_key
        self.private_key = private_key
        self.prp_key = prp_key
        self.conceal_feature = conceal_feature
        self.decrypt_count = 0  # instrumentation for decryption-workload experiments

    def tag_feature_vector(self, feature_vector):
        """
        Encrypt each feature value under the shared Paillier public
        key, tagged the same way the rule store tags its rules.

        Returns:
            dict {feature_tag: ciphertext}
        """
        tagged = {}
        for feature_index, value in enumerate(feature_vector):
            ct = paillier.encrypt(encode(value), self.public_key)
            tag = prp_int(self.prp_key, feature_index) if self.conceal_feature else feature_index
            tagged[tag] = ct
        return tagged

    def _recover_direction(self, blinded_ct):
        """
        Decrypt a blinded difference ciphertext and recover only its
        sign, using the centered representation of Z_n: a decrypted
        value greater than n/2 represents a negative number.
        """
        self.decrypt_count += 1
        n, _g = self.public_key
        m = paillier.decrypt(blinded_ct, self.public_key, self.private_key)
        centered = m if m <= n // 2 else m - n
        return "left" if centered <= 0 else "right"

    def classify(self, cloud, feature_vector, paths):
        """
        Run the full two-round protocol against `cloud` and return the
        predicted label.

        Evaluates paths in declaration order and returns the label of
        the first path whose every condition's recovered sign matches
        its required direction.
        """
        tagged = self.tag_feature_vector(feature_vector)
        blinded = cloud.evaluate_round(tagged)

        matches_by_path = defaultdict(list)
        for rule, blinded_ct in zip(cloud.rule_store.rules, blinded):
            recovered = self._recover_direction(blinded_ct)
            matches_by_path[rule.path_id].append(recovered == rule.direction)

        for path in paths:
            conditions_matched = matches_by_path.get(path.path_id, [])
            if len(conditions_matched) == len(path.conditions) and all(conditions_matched):
                return path.label
        return None
