# PrivPathInfer (static)

Companion code for *"PrivPathInfer: Continuous Fidelity, Tunable
Storage-Linkability, and Reduced Decryption Workload for
Privacy-Preserving Decision Tree Inference in Cloud-Assisted Medical
Diagnosis."*

This repository implements fixed-point-encoded Paillier comparison, a
tunable storage/linkability deduplication parameter, and a
router/subtree split that reduces the User's decryption workload. The
rule store is written once from a fixed decision tree and never
modified afterward: no class in `system/` or `crypto/` has a
delete/modify/revoke method for changing a rule after it is written,
and no data structure carries a version field or change-token for
that purpose. `tests/test_all.py` and the experiment scripts exercise
only this write-once, single-shot inference path.

## Layout

- `crypto/` — Paillier with gmpy2-accelerated modular exponentiation
  and a pure-Python fallback; `crypto/aes128.py` and
  `crypto/prf_prp.py` provide AES-128 and the PRF/PRP built on it,
  used to conceal feature identity in the rule store.
- `system/` — path extraction, the deduplication-parameter rule store,
  the two-round secure comparison protocol (`CloudParty`/`UserParty`),
  and the router/subtree partitioner.
- `baseline/` — an SDTC implementation (Liang et al. 2021), used only
  as a comparison point for the storage and fidelity evaluation.
- `experiments/` — dataset loaders and the four experiments (fidelity,
  storage/dedup, subtree decryption workload, SDTC baseline).
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
python -m experiments.exp_sdtc_baseline
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
  tree depths 2-12 at `c=1` vs `c="max"`, plus an SDTC (5-bin) series
  at the same depths for comparison, and a deduplication ablation at
  depth 8 across `c in {1,2,4,8,16,32,"max"}` (ciphertext count,
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
- **`exp_sdtc_baseline`** — SDTC's own classification disagreement
  rate against the plaintext tree at 5-bin discretization, across all
  four datasets. This is SDTC's accuracy-loss number, not
  PrivPathInfer's (PrivPathInfer's own agreement is `exp_fidelity`,
  and it does not discretize at all).

## Storage byte-size methodology

`system/rule_store.py:measure_storage_bytes` reports storage as: one
copy of each **distinct** Paillier ciphertext, plus a fixed-size
bookkeeping payload for **every** rule row (ciphertexts may be shared
across rows when `c > 1`; bookkeeping is per row regardless).

- **Ciphertext size**: a Paillier ciphertext is an element of
  `Z_{n^2}`, so its serialized size is `2 x bit_length(n)` bits, i.e.
  `2 x bit_length(n) / 8` bytes — 256 bytes at a 1024-bit `n`.
- **Per-rule bookkeeping**: a feature tag (16 bytes when PRP-concealed
  at `c=1`, 4 bytes as a plaintext `int32` index when `c>1`), a
  1-byte direction, a 4-byte path id, and a 1-byte label.

The SDTC comparison series in `exp_storage_depth_sweep.json` uses
SDTC's own natural unit: each decision-table entry is two 16-byte
PRF/PRP outputs (`encrypted_key`, `encrypted_label` — 32 bytes/entry),
with one entry per distinct discretized training sample (the
data-driven "comparing method" table construction, Section 4.3 of
Liang et al. 2021), not one per root-to-leaf path — so, unlike
PrivPathInfer's own series, this SDTC series does not vary with tree
depth.

This is a from-scratch accounting choice, not a reproduction of any
other paper's byte model — if these storage numbers don't match
another PrivPathInfer-related paper's figures, this is why, and the
per-field sizes above are the answer to give a reviewer who asks.
