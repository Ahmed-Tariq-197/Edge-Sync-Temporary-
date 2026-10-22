# TASK7_WEIGHTS_PROVENANCE — ES-EXP07 Task 7 model assets

**Scope:** every weight, projection and exported graph that a Task 7 number touches, inventoried by SHA-256 and by measured internal structure.
**Written:** 2026-10-19 · host as in `TASK7_DATA_PROVENANCE.md`.
**Rule applied throughout:** parameter counts are counted from the tensors, never inferred from file size. `414,724,314 B` is a file size; `100,769,512` is the deployed arm's trainable parameter count; the two are not interchangeable and are reported separately everywhere below.
Labels: `COMPUTED` (measured here), `VERIFIED-BY-HASH`, `NOT RECORDED`, `BLOCKED`.

---

## W-01 TransReID backbone checkpoint — the real weights

| Field | Measured value |
|---|---|
| Path | `Implementation Phase\edgesync_ws\src\edgesync_bringup\models\transreid\transformer_120.pth` |
| Size | **414,724,314 B** (`VERIFIED-BY-HASH` at every gate; re-hashed today) |
| SHA-256 | `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5` |
| Container | flat `OrderedDict`, **no** `state_dict`/`epoch`/`args` wrapper — `COMPUTED` |
| Tensor entries | **211** |
| Total elements in the file | **103,661,037** |
| Architecture implied by the tensors | ViT-B/16: `pos_embed (1, 211, 768)`, `sie_embed (6, 1, 768)`, 12 `base.blocks.*`, `patch_embed.proj.weight (768, 3, 16, 16)`, BNNeck `bottleneck (768)` — `COMPUTED` |
| Identity head in the file | `classifier.weight (751, 768)` → the checkpoint was trained on **751** identities, matching Market-1501's training split (D-01) |
| Auxiliary heads in the file | `classifier_1..4.weight (751, 768)` and `b1.*`, `b2.*` side-branch blocks, `bottleneck_1..4` — 58 non-`base.*` entries totalling **17,078,021** elements; `base.*` alone is **86,583,016** |
| What the deployed extractor loads | 210 entries after dropping `classifier.weight`; **`missing_keys == []`**, `unexpected_keys == ['classifier_1..4.weight']`, coverage **100.0 %**, model entries **206** |
| Trainable parameters of the loaded model | **100,769,512** (`sum(p.numel() for p in model.parameters())`) |
| Reconciles exactly | 103,661,037 − 5 × 576,768 (the five 751×768 heads) − 5 × 1,537 (five BNNeck buffers: running_mean + running_var + num_batches_tracked) = **100,769,512** — `COMPUTED`, so the count is derived, not quoted |
| Load path used | `create_extractor('transreid', …)` in `edgesync_identity/models/transreid_extractor.py`; the checkpoint is loaded by the production class, not by an ad-hoc script |
| Not retrained | zero training runs were executed in this task (charter §5) |

**Provenance limits, stated not hidden:** the file name `transformer_120.pth` asserts an epoch-120 checkpoint, and the tree contains **no download record, URL, license note or upstream checksum** for it — `NOT RECORDED`. What *is* verifiable on this host is internal: 211 tensors, 100 % coverage against the shipped architecture, the 751-identity head, and the SIE table with 6 camera embeddings. Authorship upstream of the project is not something this audit can establish.

## W-02 The 768 → 512 projection (fitted artefact — consumed as shipped, never refit)

Directory: `…\edgesync_bringup\models\transreid\` (and duplicated under `…\models\projections\`, see W-05).

| File | Bytes | SHA-256 (16) | Keys inside | Retained variance R₅₁₂ |
|---|---|---|---|---|
| **`svd_projection_768_to_512_real_v2.npz`** — **deployed** | **1,466,587** | **`9626bea5526df5b0`** | `projection (768,512) f32`, `mean (768) f32`, `singular_values (512)`, `total_variance_ratio` | **0.9864118** |
| `svd_projection_768_to_512_real_v3_reproduce.npz` | 1,466,587 | `9626bea5526df5b0` | identical | 0.9864118 |
| `svd_projection_768_to_512_real.npz` | 1,466,710 | `dc2b9072850c299d` | same four keys | 0.9857139 |
| `svd_projection_768_to_512.npz` | 1,466,710 | `dc2b9072850c299d` | same | 0.9857139 |
| `svd_projection_768_to_512_norm05_centered.npz` | 1,464,564 | `c9bca7595cb67d63` | `projection`, `mean`, `target_dim`, `normalization` | not stored |
| `svd_projection_768_to_512_norm05.npz` | 1,461,626 | `628b1dc6bcec1c22` | `projection`, `mean (all zeros)`, `target_dim`, `normalization` | not stored |

Measured linear algebra of the **deployed** `real_v2` (`COMPUTED`, from the tensors themselves — the `.npz` stores no recipe metadata):

- recipe is **centred** truncated SVD: `mean` has L2 norm **10.372914** (‖μ‖∞ element −4.8736), so the transform is `x → (x − μ) P` rather than a raw projection;
- `Pᵀ P = I₅₁₂` to **max abs 9e-09** → the 512 columns are orthonormal, so the map is an isometry onto its range and cosine similarity on the projected vector is well-defined;
- singular values span **537.587 → 25.712** over the 512 retained components; **R₅₁₂ = 0.9864118** (≥ the script's 0.95 gate);
- applied order at inference (documented contract, asserted per image): centre → SVD → finite check → L² normalise.

Relationships between the variants, measured rather than inferred from names:

- `real_v2` ≡ `real_v3_reproduce` — **byte-identical**, so a documented reproduction of the fit exists and returns the identical matrix.
- `svd_projection_768_to_512.npz` ≡ `svd_projection_768_to_512_real.npz` — **byte-identical** (`dc2b9072…`). The suffix "real" therefore does **not** distinguish this file from the earlier one; it differs from the deployed `real_v2` by up to **0.2518** per element. Anyone selecting an asset by name alone can pick the wrong matrix.
- `norm05` is **non-centred** (its `mean` is all zeros) and differs from `real_v2` by up to 0.4299; `norm05_centered` matches `real_v2`'s recipe (same ‖μ‖, same mean min) but not its values — max abs difference **3.4235e-05** — so `real_v2` is a later refit of that recipe, not a copy of it.

**Fitting population — the claim versus what the data shows.** `fit_svd_projection.py:17-21` documents "SVD Fitting: IDs 0001–0650", and `--id-min/--id-max` default to 1/650, selecting `bounding_box_train/` images. Measured against the actual files (D-01/D-02): IDs 1–650 exist for only **341** of the 751 training identities, covering **6,241** images; the "calibration" band 651–751 is **55** identities / 1,180 images. So the real fitting population is 341 identities / 6,241 train-split images, and the documented band numbers describe neither how Market-1501 is split nor how many identities the band captures.

Two consequences, both important:

1. **No test leakage.** That fitting population lies entirely inside the *training* split, which is disjoint by identity number from every gallery and query row used in Task 7 (measured `train ∩ test = 0`, `train ∩ query = 0`). The projection was not fit on evaluation data.
2. **The variance figure is in-sample.** R₅₁₂ = 0.9864 is measured on features drawn from the same checkpoint that produced the features being projected, on the split that checkpoint was trained on. It is a property of that fit, not an out-of-sample compression claim.

**Why this asset was not refit here:** charter §5 forbids silently refitting the projection or changing its training population. Beyond that, the guard is *measurable*: the deployed `.npz` stores `projection`, `mean`, `singular_values`, `total_variance_ratio`, while today's `fit_svd_projection.py` additionally writes `explained_variance_ratio` — a key that is **absent** from the shipped file. The deployed matrix therefore came from an earlier version of that script, and re-running the current script is *not* a proven way to reproduce `9626bea5…`. Task 7 consumed the shipped file, hash-pinned: `svd_projection_768_to_512_real_v2.npz`, 1,466,587 B, `9626bea5526df5b0a9d5b4dccec1c552224b5a6da504a0ab3a69f6bbfec1dbfc`, recorded in the embeddings manifest and re-verified by the statistics consumer before use.

## W-03 Exported ONNX graphs (charter §6 steps 1–5) and the TensorRT status

| Artefact | Bytes | SHA-256 (16) | Notes measured at export |
|---|---|---|---|
| `transreid_feature_agnostic.onnx` | 343,380,183 | `318b6857585d5d7c` | opset `ai.onnx:17`, ir_version 8, **779 nodes**, 154 initializers ≈ 343.25 MB f32, input `image (1,3,256,128) f32`, output 768-D, symbolic batch |
| `transreid_feature_sie.onnx` | 343,399,114 | `2cd8ccc774c59337` | same, **783 nodes**, 155 initializers, extra input `cam_label int64` |

Both exported **from the production extractor's own `_model`**; the SVD projection stays outside the graph (charter §5 — a fitted artefact fused into a binary engine is unauditable). `onnx.checker` PASS ×2, off-contract `128×64` input refused by the graph, Torch↔ORT fp32 parity worst max-abs **1.2397766e-05** (768-D) / **6.090850e-07** (512-D post-contract), cosine ≥ 0.999999999995. Re-exported into a separate directory and **byte-identical** on both graphs. Location: `C:\Users\PC\edgesync_es07_runs\onnx_trt09\` and `…\onnx_trt09_rerun\` (outside the repo).

**TensorRT engine: `BLOCKED`.** No engine exists — `tensorrt` and `pycuda` are unimportable on this host and none of the Jetson identity files is present. `scripts/build_transreid_trt_engine.py` was executed here and exited **1 by design**, writing `probe_build_report.json` with the environment record and the verdict `BLOCKED — Pending Jetson Orin Nano access`. No deployment, latency or on-device parity number is claimed from these files.

## W-04 Synthetic fixture — must never be read as weights

`Implementation Phase\edgesync_ws\src\edgesync_identity\tests\fixtures\synthetic_transreid.pth`, **414,718,880 B**, SHA-256 starts `2e518f28bb5ee501`. Its size deliberately mimics the real checkpoint (it differs by 5,434 B), so **file size cannot be used to tell them apart** — that is precisely why every artifact in this document is pinned by hash. It exists to exercise the load/guard paths of `test_transreid_extractor_guardrails.py` cheaply; no Task 7 embedding came from it (all 19,281 rows trace to `843055293e…`, W-01).

## W-05 OSNet baseline assets — inspected only (charter §3B: no repair)

| Asset | Measured |
|---|---|
| `…\models\osnet\osnet_x1_0_market1501.pth` | 10,399,605 B, SHA-256 `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`; flat `OrderedDict`, **567** tensor entries, **2,577,891** elements = **2,554,771 trainable + 23,120 BN buffers**; `classifier.weight (751, 512)` → 751 training identities, the same split TransReID used |
| `models/projections/svd_projection_768_to_512.npz`, `…_norm05_centered.npz` | byte-identical duplicates of the `models/transreid/` copies (`dc2b9072…`, `c9bca7595c…`) — two paths, one matrix each; neither is the deployed asset |

**Findings recorded, nothing repaired.** `osnet_torch_extractor.py:77-82, 93-107` falls back (on `ImportError` of `torchreid`) to `_build_osnet_standalone()`, which the code itself calls a "Minimal conv backbone placeholder". Measured by instantiating that method only (read-only, no file touched): **8 state-dict entries, 42,945 elements = 42,816 trainable + 129 buffers**, one `Conv2d(3→64, 7×7)` plus `Linear(64→512)`. Against the 567-key checkpoint, `load_state_dict(..., strict=False)` reports **missing 7, unexpected 567, overlap 0 of 567** — i.e. *not one* checkpoint tensor is applied and the arm returns features from randomly-initialised weights while logging `params=42,816`. On this host that branch is not even reached: `import torchreid` raises `RuntimeError: operator torchvision::nms does not exist` (a `RuntimeError`, not an `ImportError`), which is why the OSNet arm produced **no embeddings** and the OSNet↔TransReID paired test is `NOT APPLICABLE` here. Team-owned tests `test_experiment7_fixes.py` / `test_experiment7_forensic_fixes.py` pass against this placeholder, so their green status is not evidence of a working OSNet arm. Charter §3B: reported, not fixed.

## W-06 What the Task 7 numbers depend on, in one list

| Dependency | Pin used by the run | Verified before use |
|---|---|---|
| Backbone weights | `transformer_120.pth` `843055293e160da0…`, 414,724,314 B | manifest asset block + consumer integrity gate |
| Projection | `svd_projection_768_to_512_real_v2.npz` `9626bea5526df5b0…`, 1,466,587 B | same |
| Threshold | τ = 0.70 from `algorithm_thresholds.yaml` key `…reid_similarity_threshold` | read at run time; `matches_expected_0.70: true`; a different value aborts |
| Embeddings | three `.npz` files, digests in `TASK7_DATA_PROVENANCE.md` D-04 | SHA-256 re-hashed by the consumer, `match: true` ×3 |
| Metric implementation | `run_real_pipeline.compute_cmc_map` via `evaluate_offline` | per-query pass asserted equal; max abs difference exactly 0.0 |
| Similarity | `core/identity_verifier.py` float32 cosine, accept iff `sim >= τ` | cross-checked on 200 real pairs, 1.788e-07 |

## W-07 Reproducing the measurements in this document

`python C:\Users\PC\edgesync_es07_runs\provenance_measure.py` (**exit 0**, ~8 s) emits every number
quoted above into
`C:\Users\PC\edgesync_es07_runs\provenance\task7_provenance_measurements.json`: the two checkpoint
inventories (entry counts, element sums, buffer split, head shapes, the 100,769,512 reconciliation), all
six projection digests with `mean_l2`, `columns_orthonormal_max_abs_deviation`,
`singular_value_first/last` and `total_variance_ratio`, the pairwise `byte_identical` /
`max_abs_element_difference` table, and the OSNet placeholder instantiated in memory with its
`key_overlap_with_checkpoint: 0` / `strict_false_missing_keys: 7` /
`strict_false_unexpected_keys: 567`. It writes nothing inside the project tree; OSNet is read and
instantiated only, never modified (§3B).
