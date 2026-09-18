# PrivPathInfer (static)

Companion code for *"PrivPathInfer: Continuous Fidelity, Tunable
Storage-Linkability, and Reduced Decryption Workload for
Privacy-Preserving Decision Tree Inference in Cloud-Assisted Medical
Diagnosis."*

This repository implements a **subset** of a larger thesis system:
fixed-point-encoded Paillier comparison, a tunable
storage/linkability deduplication parameter, and a router/subtree
split that reduces the User's decryption workload. It is deliberately
**static**: there is no rule-update, deletion, or versioning capability
anywhere in this codebase, on purpose — that capability belongs to a
separate paper and would create a publication conflict if mixed in
here. This isn't just a documentation claim: no class in `system/` or
`crypto/` has a delete/update/remove/revoke method, no data structure
carries a deletion token or version field, and there is no batch or
dummy-padding protocol. `tests/test_all.py` and the experiment scripts
exercise only static, single-shot inference.

## Layout

- `crypto/` — Paillier (gmpy2-accelerated modexp with a pure-Python
  fallback), AES-128, and a PRF/PRP built on it. Copied from the
  original thesis repository's crypto layer and patched only for
  performance; `crypto/prf_prp.py` additionally had its
  update-protocol token functions (`generate_deletion_token`,
  `verify_deletion_token`, `derive_encryption_key`) removed, since the
  upstream file — despite the thesis's own crypto layer being update-
  agnostic in principle — included them.
- `system/` — path extraction, the deduplication-parameter rule store,
  the two-round secure comparison protocol (`CloudParty`/`UserParty`),
  and the router/subtree partitioner.
- `experiments/` — dataset loaders and the three experiments
  (fidelity, storage/dedup, subtree decryption workload).
- `data/` — raw dataset files (`diabetes.csv`, `processed.cleveland.data`,
  `framingham.csv`); `breast_cancer` loads directly from scikit-learn.
- `tests/test_all.py` — encoding, Paillier correctness/homomorphism,
  the dedup construction, and the two-round protocol / subtree
  partitioner against plaintext on a synthetic tree.
- `results/` — JSON output of each experiment script.

## Running

```
pip install -r requirements.txt
python -m tests.test_all
python -m experiments.exp_fidelity
python -m experiments.exp_storage_dedup
python -m experiments.exp_subtree_workload
```

All experiments use 1024-bit Paillier keys throughout (gmpy2 required
for this to be tractable; falls back to `pow` if gmpy2 isn't
installed, which will be much slower at this key size).

## What each experiment reports

- **`exp_fidelity`** — per-dataset classification agreement between
  the plaintext tree and the full secure protocol, over every record.
  This is a single full pass per dataset rather than ten: the sign
  recovered from `Enc(r * (feature - threshold))` is exact for any
  blinding scalar `r` as long as the true difference stays far inside
  `(-n/2, n/2)`, which holds by an enormous margin at a 1024-bit key.
  There is no run-to-run variance to average over (this is exercised
  directly in `tests/test_all.py`).
- **`exp_storage_dedup`** — PIMA rule-store storage size (KB) across
  tree depths 2-12 at `c=1` vs `c="max"`, and a deduplication ablation
  at depth 8 across `c in {1,2,4,8,16,32,"max"}` (ciphertext count,
  storage, linkability). Ten repetitions per configuration; storage is
  reported as mean/std, ciphertext count and linkability are exact.
- **`exp_subtree_workload`** — User-side decryption time under
  `baseline_classify` (no partitioning) vs `subtree_classify` at
  router depths 1-4, averaged over 10 queries per configuration.
  Reports only the within-scheme reduction percentage and the
  underlying rule/decryption counts — never a cross-scheme or
  plaintext timing comparison, since Paillier's per-operation cost is
  expected to be far slower than plaintext and is not a contribution
  of this work.
