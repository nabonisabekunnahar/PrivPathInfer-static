"""
sdtc.py — SDTC baseline implementation, used only as a storage/fidelity
comparison point for this paper's evaluation.

Reference:
    Liang, J., Qin, Z., Xiao, S., Ou, L., and Lin, X.
    "Efficient and Secure Decision Tree Classification for
    Cloud-Assisted Online Diagnosis Services."
    IEEE Transactions on Dependable and Secure Computing,
    Vol. 18, No. 4, July/August 2021.

SDTC Scheme Overview (the "comparing method", Section 4.3):
    1. Discretize all continuous features into k bins.
    2. Build a decision table keyed by the FULL discretized feature
       vector of every training sample (deduplicated): S[i] -> label,
       where label is the tree's prediction at that bin combination's
       representative point (bin midpoints). This is data-driven
       rather than enumerating all k^n combinations, since k^n is
       intractable for realistic feature counts.
    3. Encrypt each row's key and label with a PRF/PRP.
    4. Inference: discretize the query the same way, derive the same
       key, and look it up.

The multi-feature key is encoded as a single mixed-radix integer
S = sum_i(bin_i * k^i) rather than packed one-byte-per-feature, since
byte-packing silently truncates past 16 features (this dataset set
includes a 30-feature one); the mixed-radix integer stays within a
16-byte PRF/PRP block for the bin counts used here.

Limitations this paper's evaluation compares against:
    Accuracy loss — discretization causes misclassification whenever a
    tree threshold falls strictly inside a bin; PrivPathInfer avoids
    this by encrypting exact fixed-point-encoded thresholds instead of
    bin indices.

    Storage — the decision table has one entry per distinct discretized
    training sample, with no cross-entry sharing; PrivPathInfer's rule
    store shares ciphertexts across paths via the c parameter.
"""

import numpy as np
from typing import Dict, Optional

from crypto.prf_prp import prf, prp
from baseline.discretizer import Discretizer

import os

ENTRY_BYTES = 16 * 2  # encrypted_key + encrypted_label


def _boolean_string(disc_features: np.ndarray, n_bins: int) -> int:
    """Encode a discretized feature vector as one mixed-radix integer:
    S = b0*n_bins^(n-1) + b1*n_bins^(n-2) + ... + b_{n-1}."""
    result = 0
    for val in disc_features:
        result = result * n_bins + int(val)
    return result


class SDTC:
    """
    SDTC: Secure Decision Tree Classification baseline (Liang et al.
    2021), used here only for comparison.

    Key differences from PrivPathInfer:
        - Requires discretization (accuracy loss)
        - One table entry per distinct discretized training sample, no
          cross-entry sharing
        - This reference implementation does a dict lookup by
          encrypted key; the scheme's own O(1) claim relies on an
          SSE-style array/XOR indirection this reference skips since
          it doesn't change the storage or accuracy figures compared
          here
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins      = n_bins
        self.discretizer = Discretizer(n_bins=n_bins, strategy='equal_width')

        self.K1 = os.urandom(16)  # PRF key for the table key
        self.K2 = os.urandom(16)  # PRP key for the encrypted label

        self.table: Dict[bytes, bytes] = {}     # encrypted_key -> encrypted_label
        self._label_map: Dict[bytes, int] = {}  # encrypted_label -> label
        self.fitted = False

    def fit_encrypt(self, sklearn_tree, X_train: np.ndarray):
        """
        Fit the discretizer, build the decision table from the
        distinct discretized training samples (comparing method,
        Section 4.3), and encrypt every row.
        """
        self.discretizer.fit(X_train)
        X_disc = self.discretizer.transform(X_train)
        bin_edges = self.discretizer.bin_edges_per_feature

        midpoints_per_feature = []
        for edges in bin_edges:
            mids = [(edges[j] + edges[j + 1]) / 2.0 for j in range(len(edges) - 1)]
            mids = [edges[0]] + mids + [edges[-1]]
            midpoints_per_feature.append(mids)

        n_features = X_train.shape[1]
        self.table = {}
        self._label_map = {}
        seen = set()

        for sample_disc in X_disc:
            key = tuple(int(v) for v in sample_disc)
            if key in seen:
                continue
            seen.add(key)

            representative = np.zeros(n_features)
            for f_idx in range(n_features):
                bin_idx = max(0, min(int(sample_disc[f_idx]), len(midpoints_per_feature[f_idx]) - 1))
                representative[f_idx] = midpoints_per_feature[f_idx][bin_idx]

            label = int(sklearn_tree.predict(representative.reshape(1, -1))[0])
            self._add_entry(sample_disc, label)

        self.fitted = True

    def _add_entry(self, disc_features: np.ndarray, label: int):
        s_i = _boolean_string(disc_features, self.n_bins)
        enc_key   = prf(self.K1, s_i)
        enc_label = prp(self.K2, label)
        self.table[enc_key] = enc_label
        self._label_map[enc_label] = label

    def get_storage_size(self) -> int:
        """Number of entries in the encrypted decision table."""
        return len(self.table)

    def get_storage_bytes(self) -> int:
        """Serialized byte size of the encrypted decision table."""
        return len(self.table) * ENTRY_BYTES

    def classify_disc(self, disc_features: np.ndarray) -> Optional[int]:
        """Classify a pre-discretized feature vector via table lookup."""
        assert self.fitted, "Call fit_encrypt() first"
        s_i = _boolean_string(disc_features, self.n_bins)
        enc_key = prf(self.K1, s_i)
        enc_label = self.table.get(enc_key)
        if enc_label is None:
            return None
        return self._label_map.get(enc_label)

    def classify(self, features: np.ndarray) -> Optional[int]:
        """Classify a continuous feature vector (discretize then look up)."""
        disc = self.discretizer.transform(np.array(features).reshape(1, -1))[0]
        return self.classify_disc(disc)


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("SDTC Baseline Verification Tests")
    print("Reference: Liang et al. 2021, IEEE TDSC")
    print("=" * 60)

    X = np.array([
        [50.0,  85.0],
        [120.0, 126.5],
        [160.0, 150.0],
    ], dtype=float)

    disc = Discretizer(n_bins=10)
    X_disc = disc.fit_transform(X)
    assert X_disc.shape == X.shape
    print("[PASS] Test 1: Discretizer integration")

    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(0)
    X_train = rng.rand(100, 4) * 200
    y_train = (X_train[:, 0] > 100).astype(int)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    sdtc = SDTC(n_bins=5)
    sdtc.fit_encrypt(clf, X_train)
    assert sdtc.get_storage_size() > 0, "Table should have entries"
    print(f"[PASS] Test 2: SDTC table encrypted ({sdtc.get_storage_size()} entries)")

    # Test 3: a query identical to a training sample should round-trip
    # to that sample's own predicted label — the whole point of the
    # data-driven table construction.
    plaintext_preds = clf.predict(X_train)
    matches = sum(
        1 for i in range(len(X_train))
        if sdtc.classify(X_train[i]) == int(plaintext_preds[i])
    )
    agreement_pct = 100.0 * matches / len(X_train)
    assert agreement_pct > 50.0, (
        f"Data-driven table should agree with the tree far more than "
        f"chance on its own training data; got {agreement_pct:.1f}%"
    )
    print(f"[PASS] Test 3: table lookups agree with plaintext on {agreement_pct:.1f}% of training samples")

    print("\n[ALL TESTS PASSED] sdtc.py verified.")


if __name__ == "__main__":
    run_all_tests()
