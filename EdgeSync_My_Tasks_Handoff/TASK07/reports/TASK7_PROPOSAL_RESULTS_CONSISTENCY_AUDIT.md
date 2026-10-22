# Task 7 — Proposal ↔ recorded-results consistency audit (ES-EXP07)

Status: **COMPLETE for the scope audited.** Owner: Abanob. Method: **read-only**. No file under the project tree was modified by this audit; every number below is either quoted from a named artifact with its measured SHA-256, or computed from tensors/bytes read in this session. Nothing here repairs OSNet (charter §3B) or re-runs training, projection fitting, ONNX export or TensorRT.

**Scope.** Charter §11 asks whether the Task 7 proposal (`ES-EXP07-TASK7-PROPOSAL-01-REV01.md`, `…-REV02.md`) and the results already on disk (`results/experiment7/*.json`, plus the two reproductions run on 2026-10-19) say the same thing. Verdict per claim, with the contradiction classed rather than smoothed over.

**Labels (charter §8).** `COMPUTED` = measured in this session from bytes on disk · `REPRODUCED` = matches a prior measurement to the digit · `QUOTED` = copied verbatim from a named artifact · `UNVERIFIED` = asserted somewhere on disk with no source reachable · `NOT MEASURED` = no such measurement exists on this host · `NOT APPLICABLE` = the comparison is not defined.

**Environment for every `COMPUTED` row.** Windows 10.0.26300 x64, Python 3.10.11, torch 2.5.1+cu121, numpy 1.26.4, RTX 3050 Laptop 4 GiB. `MiB` below = bytes / 2²⁰; artifacts that print "MB" are examined in both readings and the discrepancy is called out where it matters.

---

## 1. Top-line verdict

| # | Question | Verdict |
|---|---|---|
| 1 | Do the proposal's accuracy claims match the recorded accuracy artifacts? | **YES for the candidate arm** — `transreid_512_sie` **reproduces** the recorded row digit-for-digit (CA-01). The proposal's *reason* for refusing that comparison was **false** (CA-02) |
| 2 | Do the proposal's memory/size claims match the artifacts? | **YES, and now exactly derivable** — all four recorded size figures reconstruct from measured element counts (CA-05). The "40.5×" label is wrong; the ratio it names is 46.4× (CA-06) |
| 3 | Do the proposal's latency claims match the artifacts? | **NO — irreconcilable in sign** across three artifacts, and the proposal says so; two of its own cited sources are absent from disk (CA-07, CA-08) |
| 4 | Is the τ = 0.70 operating point consistent between proposal and artifacts? | **NO** — the proposal's τ provenance is sound (CA-09), but the recorded FNMR/FMR/EER artifact declares **no** pair population, so no recorded verification number is comparable (CA-10) |
| 5 | Are the artifacts internally consistent with one another? | **NO** — four independent contradictions: CA-03 (population excuse), CA-04 (two mAP implementations), CA-11 (three values for one named arm), CA-12 (the peak-memory figure ranks the *smaller* model higher) |
| 6 | Is the recorded production recommendation supportable? | **NO** — `gate3_real_results_reconciled.json` recommends rejecting TransReID on the same page that records Jetson as `NOT MEASURED` (CA-13). The proposal and report both refuse to repeat the pattern; that refusal is the only device-adjacent claim in this experiment that holds |
| 7 | Does the evidence change the hypothesis verdicts? | **No.** H7a stays `NOT VERIFIED BY MEASUREMENT HERE`; H7b/H7c stay `NOT MEASURED`. What changes is that the accuracy half is now a **reproduction**, which is stronger than "a number on an unknown set" (CA-01) |

---

## 2. Artifacts audited (measured digests)

All in `Implementation Phase/edgesync_ws/src/edgesync_bringup/results/experiment7/`. `COMPUTED`.

| File | Bytes | SHA-256 (16) | Generator script on disk? |
|---|---|---|---|
| `gate3_real_results.json` | 1,557 | `fe42f9885db05c80` | **NO** |
| `gate3_real_results_v2_corrected.json` | 3,247 | `81c90c3b45b0e9e0` | **YES** — `edgesync_identity/scripts/run_real_pipeline.py:40` writes exactly this path |
| `gate3_real_results_v3_reproduce.json` | 1,667 | `915f55547649ed20` | **NO** |
| `gate3_real_results_reconciled.json` | 2,263 | `564323622381700a` | **NO** |
| `forensic_ablation_results.json` | 2,180 | `a283f66d554c9b94` | **NO** |
| `verification_threshold_audit.json` | 1,542 | `f80fd7b9ebd941d0` | **NO** |
| `benchmark_latency_memory.json` | 1,125 | `6becced684fa1493` | **NO** |

`COMPUTED` by grep over `Implementation Phase/edgesync_ws/src/**/*.py`: only two of the seven filenames are referenced by any script, and one of those references is a *reader* (`evaluate_offline.py:80` cites `gate3_real_results_v2_corrected.json` as the population provenance), so **five of seven recorded result files have no producer in this tree**. `UNVERIFIED CLAIM` is therefore the correct status for their methods, even where their numbers are internally consistent.

Additional evidence files read: `docs/experiments/exp07/audit/evidence/repro_transreid_both_arms.json` (team reproduction, 2026-10-19), `docs/experiments/exp07/audit/ES-EXP07-AUDIT-FORENSIC-01.md`, `docs/experiments/exp07/audit/artifact-inventory.md`, `docs/experiments/exp07/audit/audit_20261019_141628/model_asset_inventory.md`, and this pass's own outputs `C:\Users\PC\edgesync_es07_runs\statistics\task7_statistics.json` (`76d88dfc5e5c20c4`), `…\embeddings_full_directory\task7_embeddings_manifest.json` (`283c8c782e66869a`), `…\provenance\task7_provenance_measurements.json`.

---

## 3. Accuracy: proposal ↔ record

### CA-01 — The candidate arm **reproduces** the record. `COMPUTED` / `REPRODUCED`

| Source | Rank-1 % | Rank-5 % | Rank-10 % | mAP % | n_query | n_gallery |
|---|---|---|---|---|---|---|
| `gate3_real_results_v2_corrected.json` | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | 85.12520406484964 | 3,368 | 15,913 |
| `gate3_real_results_v3_reproduce.json` | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | 85.12520406484964 | 3,368 | 15,913 |
| team reproduction, `WITH_CAM_SIE` (`repro_transreid_both_arms.json`) | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | 85.12520406484964 | 3,368 | 15,913 |
| **this pass, `transreid_512_sie`** | **93.37885985748218** | **97.95130641330167** | **98.87173396674585** | **85.12520406508766** | 3,368 | 15,913 |
| this pass, `transreid_512_agnostic` | 88.74703087885986 | 96.14014251781472 | 97.8622327790974 | 77.32400426226894 | 3,368 | 15,913 |
| team reproduction, `NO_CAM_deployed_path` | 88.74703087885986 | 96.14014251781472 | 97.8622327790974 | 77.32400426241038 | 3,368 | 15,913 |

Rank-1/5/10 are **identical to the last printed digit** in four independent places for the SIE arm and in two for the agnostic arm. mAP differs between the record and this pass by **+2.3802e-10** (SIE) and **−1.4144e-10** (agnostic) — relative 2.8e-12 and 1.8e-12, i.e. float accumulation order, not a protocol difference. `svd_variance_retained` in both gate3 v2/v3 = 0.9864118099212646, matching the deployed projection's measured R₅₁₂ = 0.9864118 (§CA-14).

**Consequence.** The proposal REV01 §2/§5 statement that the recorded TransReID row is "a **different population**" is falsified (CA-02), and the report's `§4.3` label `REPRODUCED` is justified. This is the strongest pipeline-correctness evidence available without the device: the extractor, `[0.5]³` normalisation, post-BNNeck CLS head, SIE path, the pinned checkpoint `843055293e160da0…` and the pinned projection `9626bea5…` together regenerate a two-year-old recorded result exactly.

**What it does not license.** No paired OSNet-vs-TransReID test: `accuracy/osnet` in those files is an aggregate with **no per-query outcomes anywhere on disk**, and no runnable OSNet extractor exists on this host (CA-16). Charter §8 permits a paired test only when both arms have valid paired query-level predictions. `NOT APPLICABLE` — for the baseline's sake, not the population's.

### CA-02 — The "different population" explanation is false, in three places including this pass's own tooling

`COMPUTED`. All four Gate-3 files record `n_query 3,368` / `n_gallery 15,913` (grep of every `n_query`/`n_gallery` key in the four files). The `id_band` rule — `evaluate_offline.py:113` defaults `id_min=752, id_max=1501`, applied **only** when `population='id_band_752_1501'` (`:155`) — would drop 355 of 750 test identities, and after the B-12/TR-07 fix `DEFAULT_POPULATION` is `directory` (defined at `:99`, used at `:112` and `:473`; the document-string claim is at `:7`), and the chosen name is recorded per-output (manifest field `:175`, written at `:572`), so the band is now an **opt-in, named, self-labelling** choice rather than a default. Calling it "a live defect in the default path" (as REV02 §5 item 2 does, following REV01 P5) **overstates its current reach**: it produced none of the recorded accuracy rows *and* no longer captures unwary runs. Corrected in the proposal. Where the claim is still standing on disk:

| Location | Text | Status |
|---|---|---|
| `ES-EXP07-TASK7-PROPOSAL-01-REV01.md:35` (P3 row) and `:125` | 512-D SVD arm is "a **different population**" | **FALSE** — same population, different representation. Corrected in REV02 §2/§5/§9/Appendix A |
| `analyze_task7_statistics.py:1140-1142` (this pass's consumer) | "the recorded run used the invalid id_band population … not the same population as this one" | **FALSE reason, correct conclusion.** Documented, **not patched**: editing it would require a §1 backup plus a re-run of its 68 tests for a string, and the conclusion (`NOT APPLICABLE`) does not change either way. Recorded in `ES-EXP07-TASK7-REPORT-01-REV02.md` §6 threat 10 |
| `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §5 row 26 | this finding, as measured | `COMPUTED` |

### CA-03 — `gate3_real_results.json` (71.1698) is the same population, and is *identical to a named forensic condition*

`COMPUTED`. `gate3_real_results.json → accuracy.transreid` = Rank-1 **71.16983372921615**, Rank-5 85.80760095011877, Rank-10 90.70665083135393, mAP **51.46723674145579**. `forensic_ablation_results.json` condition **D1 "TransReID 512-D SVD from Post-BN (No Cam) [ORIGINAL REPORTED PIPELINE]"** = Rank-1 **71.16983372921615**, Rank-5 85.80760095011877, Rank-10 90.70665083135393, mAP **51.46717615365994**.

So the 71.17 row was **no-camera, no-SIE, post-BNNeck, 512-D SVD** — D1 names it. The v1 → v3 change is therefore **+22.2090 pp Rank-1 / +33.6580 pp mAP** with **two** variables moved at once: SIE/camera label added (D1 → D3 is +3.414 pp Rank-1 alone: 74.58432304038006) and preprocessing/normalisation changed. `gate3_real_results.json` carries **no `pipeline_config` block at all**, so which normalisation produced 71.17 is not recorded. `gate3_real_results_reconciled.json:19` attributes the whole gap to `"Corrupted by Centered Truncated-SVD mean subtraction and omitted JPM"` while the 93.379 arm it replaced uses the **same centred** projection (‖μ‖ = 10.372914, measured) — the label contradicts the asset it condemns, and the camera embedding is not mentioned in that root-cause string although D1→D3 shows it is worth only ~3.4 pp on the forensic scale. **Verdict: the recorded attribution is incomplete and, as written, not supportable.**

### CA-04 — Two mAP implementations disagree at 6.1e-05 while agreeing exactly on Rank-1

`COMPUTED`. v1 vs forensic D1: Rank-1/5/10 identical to 14 decimals, mAP differs by **6.0588e-05** (51.46723674145579 vs 51.46717615365994, relative 1.2e-6). Same images, same features (the Rank-k equality proves it), different average-precision accumulation. This is the defect class TR-07 was; it is why `analyze_task7_statistics.py` imports the canonical `compute_cmc_map` instead of reimplementing it and asserts canonical-vs-per-query equality (measured difference **0.0** on all three arms). No recorded artifact carries that assertion.

### CA-11 — One named arm, three mutually inconsistent values on disk

| Condition "TransReID native 768-D post-BN, **no camera/SIE**" | Rank-1 % | mAP % | Source |
|---|---|---|---|
| `forensic_ablation_results.json` B1 | 70.78384798099762 | 50.281406681665075 | no generator on disk |
| `gate3_real_results_reconciled.json` `corrected_ablation_results.transreid_native_post_bn_768_no_cam` | **88.64** (2-decimal) | 74.82 | no generator; 2-decimal precision unlike every measured value in the tree |
| **this pass, `transreid_768_agnostic`** | **88.74703087885986** | 77.32735493390587 | `COMPUTED`, manifest `283c8c782e66869a`, exit 0 |

Gaps: measured − forensic = **+17.9632 pp**; measured − reconciled = **+0.1070 pp**. The reconciled figure is the only one this pass comes near, and 0.107 pp is far outside its own rounding resolution, so the two are *not* the same measurement — but they are plausibly the same arm under different feature extraction (the 768-D array here is raw post-BNNeck, deliberately not unit-norm). `forensic_ablation_results.json` cannot be reconciled at all without its preprocessing, which it does not record. **Verdict: the 768-D no-cam arm has no established value on disk.** REV02 §2 P3 already refuses to quote 88.64/89.91 as measured; this audit confirms that refusal was required.

Same pattern for the SIE effect size, recorded three ways: reconciled 768-D **+1.27 pp** (88.64 → 89.91), forensic 768-D **+3.44 / +3.47 pp** (B1→C1, B2→C2), this pass 512-D **+4.6318 pp** and mAP **+7.8012 pp** with Holm p = 4.361e-41. Only the third has a declared population, a declared pair pool, a per-query unit and a test.

---

## 4. Memory and model size

### CA-05 — Every recorded size figure reconstructs exactly from measured element counts

`COMPUTED` from `task7_provenance_measurements.json` and the files' byte lengths; `QUOTED` from the two artifacts.

| Quantity | TransReID | OSNet x1.0 |
|---|---|---|
| checkpoint file bytes | 414,724,314 (`843055293e160da0…`) | 10,399,605 (`2809d3227f7d078f…`) |
| tensor entries | 211 | 567 |
| total elements | 103,661,037 | 2,577,891 |
| buffer elements | 7,685 | 23,120 |
| **trainable elements** (total − buffers) | **103,653,352** | **2,554,771** |
| classifier-head elements | 5 × 751 × 768 = 2,883,840 | 751×512 + 751 = 385,263 |
| **trainable − heads** | **100,769,512** | **2,169,508** |
| `benchmark_latency_memory.json → parameters` | 100,769,512 ✔ exact | 2,169,508 ✔ exact |
| `parameter_mb` × 2²⁰ / 4 | 384.4051818847656 ✔ = 100,769,512×4 B | 8.276016235351562 ✔ = 2,169,508×4 B |
| `reconciled → model_size_*_mb` × 2²⁰ / 4 | 395.4061584472656 ✔ = **103,653,352**×4 B | 9.745677947998047 ✔ = **2,554,771**×4 B |
| `checkpoint_file_size_mb` | 395.511926651001 ✔ = file / 2²⁰ | 9.91783618927002 ✔ = file / 2²⁰ |
| file − trainable×4 (container + key names) | 110,906 B | 180,521 B |

**Resolution.** The two "inconsistent" memory figures in Appendix A (384.405 vs 395.512) are **not** a contradiction: one is float32 mass of the **backbone parameters**, the other is the **whole trainable state dict including the classifier heads** (or the file). Both are exactly derivable, so the Appendix A row can be quoted once the definition is named — which REV02 §9 threat 9 and report §6 threat 9 now do. Note both "MB" values are **MiB printed under an MB label**; the decimal-MB reading of the same bytes is 414.72 MB / 10.40 MB.

### CA-06 — "TransReID is 40.5x larger than OSNet" is mislabelled, and three defensible ratios exist

`COMPUTED`. `gate3_real_results_reconciled.json:62` keys this number `parameter_ratio`, but its operands (`model_size_transreid_mb` / `model_size_osnet_mb`) are the **with-heads trainable mass** from CA-05 → 40.572. The genuine parameter ratio is **46.448** (100,769,512 / 2,169,508), and the checkpoint-file ratio is **39.879**. `JETSON-RUN-SHEET-PREP-01-REV01.md:126` asserts "40.5× OSNet **in parameter mass**", which is the wrong definition for that number. No recorded artifact states which definition it used. All three are now measured, so any future quote can name its definition; **the run sheet's headline must be corrected before the device run**, because the device report is expected to restate it.

### CA-12 — The recorded "memory footprint" ranks the *smaller* model higher

`QUOTED` + `COMPUTED`. `benchmark_latency_memory.json → peak_runtime_gpu_memory_mb`: TransReID **410.7265625**, OSNet **413.26025390625** → OSNet is 2.5337 MiB **larger**. Same artifact that records 46.4× parameter mass. The peak figure is CUDA-context/allocator dominated (a 4 GiB laptop GPU), so it says nothing about model footprint; the prior audit reached the same conclusion (`ES-EXP07-AUDIT-FORENSIC-01.md`, "memory methodology is not a footprint"). Charter §7 / proposal §6 therefore require the edge figure to come from `tegrastats` MEM + `VmRSS` + `torch.cuda.max_memory_allocated()` on the Orin Nano. **Status: `NOT MEASURED`; and no laptop peak may be substituted — this artifact is the proof of why.**

---

## 5. Latency

### CA-07 — The recorded latencies disagree in sign, and the disagreement is not resolvable offline

`QUOTED` (all laptop, all nominally the same GPU class). ms.

| Artifact | TransReID median | OSNet median | TR/OS | what the artifact records about the method |
|---|---|---|---|---|
| `gate3_real_results.json` | 13.4314 | 18.8763 | **0.71×** (TR faster) | **no** warm-up/timed/iteration fields at all; note says "Laptop measurement (NOT Jetson Orin Nano 8GB)" |
| `gate3_real_results_v2_corrected.json` | 41.7151 | 28.2567 | **1.48×** (TR slower) | `warmup_iterations 50`, `timed_iterations 200`, `batch_size 1`, `includes_svd_projection true`, `projection_mean_ms 0.5822` |
| `benchmark_latency_memory.json` | 58.7181 (full pipeline) | 28.1689 (full pipeline) | **2.08×** (TR slower) | no protocol fields; **generator not on disk** |
| `gate3_real_results_reconciled.json → latency_audit` | 13.4881 (`mean_ms`) | 21.3973 (`mean_ms`) | 0.63× | **verbatim copies of v1's `mean_ms`** (byte-identical values, `COMPUTED`) — i.e. the reconciled artifact's latency row inherits the *oldest, superseded* run |
| this pass (`task7_embeddings_manifest.json`) | 15.7770 (768 and 512-agnostic, same forward) / 15.9896 (SIE) | — | n/a | self-declared: "backbone forward only", "incidental bookkeeping … without a warm-up/repeat protocol. It is **NOT** a deployment benchmark" |

Four observations that the proposal should not have to argue for:

1. **Sign flip is real.** TransReID is faster in v1 and slower in v2 and in `benchmark_latency_memory.json`. `COMPUTED` ratios above.
2. **OSNet's own "backbone" latency takes three values** — 18.8763 (v1 median), 24.7876 (benchmark backbone median), 28.1938 (v2 backbone median) — a 1.49× spread on the *same* frozen baseline. Any TransReID/OSNet ratio is therefore dominated by which artifact is picked.
3. **v2's min (14.2670) ≈ v1's median (13.4314)**, and v2's median (41.7151) is 2.9× its own min while its p99 (43.0061) sits just above the median — a warm-then-throttled distribution, not noise. So the v1-vs-v2 gap is plausibly *what is inside the timed region* plus host state, not GPU speed; but v2 also labels 38.7689 ms as TransReID's *backbone mean*, against this pass's 15.7770 ms forward-only mean for the same model — a 2.46× gap that this reading does not explain. **`UNRESOLVED`, deliberately left unresolved: resolving it requires the device harness, not more laptop runs.**
4. **This pass's own number is a fifth value** (15.7770 / 15.9896 ms) and is excluded from H7b by charter §7. Excluding it is recorded in report §8 rather than quietly dropped. The SIE-vs-agnostic cost it does show (+0.2127 ms, **+1.35 %**) is *forward-pass* cost only and is not a deployment latency.

### CA-08 — Two of Appendix A's cited latency sources are absent from disk

`COMPUTED`. Appendix A row 2 cites "`gate3_real_results.json`, `phase9`, `phase8`" for OSNet medians 18.88 / 19.74 / 20.71 and TransReID 13.43 / 13.59 / 13.84. `find` over the tree for `*phase8*` / `*phase9*` returns **nothing**; `gate3_real_results_v3_reproduce.json → latency` is the string `"not measured here -- see out/phase8_latency_*.json (idle GPU, 3 trials)"` and no `out/` directory exists under `edgesync_bringup`. **Of the six medians in that row, exactly two are traceable** (18.88 → v1 OSNet `median_ms` 18.8763; 13.43 → v1 TransReID `median_ms` 13.4314); the other four are `UNVERIFIED` — their files are not on disk. REV02 retains the row unchanged and fences it; that was the right call, and this audit supplies the missing detail: the row is not merely unreconciled, it is partly unsourced.

---

## 6. Threshold, pair populations, verification

### CA-09 — τ = 0.70 is consistent, and its provenance is now machine-checked

`COMPUTED`. `config/algorithm_thresholds.yaml` key `algorithm_thresholds.ros__parameters.reid_similarity_threshold` = 0.7; `analyze_task7_statistics.py` **refuses to run** if the loaded τ ≠ the expected 0.7 (a τ of 0.62 was tested → refused, change-log §5 row 21), and the applied rule is accept iff `sim ≥ τ`, matching `core/identity_verifier.py:101` which rejects when `sim < τ`. Recorded artifacts agree on the number: `gate3_real_results.json`, `_v2_corrected`, `_v3_reproduce`, `_reconciled` and `verification_threshold_audit.json` all carry `threshold_tau`/`tau` = 0.7. v2/v3 additionally state `"tau is not applied anywhere in this script"` — i.e. the recorded accuracy rows are threshold-free, so no recorded Rank-k implies anything about τ. **Consistent.**

### CA-10 — The recorded verification numbers are unattributable; this pass's are declared

`COMPUTED` (read the whole file). `verification_threshold_audit.json` = 1,542 B, four model rows, fields `model, genuine_mean, genuine_std, impostor_mean, impostor_std, tau, fnmr_pct, fmr_pct, eer_pct, eer_tau`. It records **no** population, pair counts, camera rule, split, gallery size, query count, script or timestamp. Its FMR values are `0.11399999999999999` and `0.013999999999999999` ×3 — the float32-ness of exactly-two-round-numbers, which implies a fixed sampled impostor count that is nowhere stated. `UNVERIFIED CLAIM`.

| Metric | Recorded (undeclared pool) | This pass (declared pool) | Comparable? |
|---|---|---|---|
| FNMR @ 0.70 | OSNet 18.703290940081292 %; TR 768+cam 42.8412 %; TR 512 centred+cam **48.6626 %**; TR 512 uncentred+cam 38.5145 % | 768-agnostic 54.6637 %; 512-agnostic 60.7384 %; **512-SIE 48.1540 %** | sign only |
| FMR @ 0.70 | 0.114 % / 0.014 % | 0.00789 % / 0.00521 % / 0.00585 % | no — pool undeclared |
| EER | 2.052 % @ 0.555 (OSNet) / 1.678 % @ 0.330 (TR centred) | 2.0095 % @ 0.36706 / 1.9718 % @ 0.30581 / **1.5242 % @ 0.32755** | sign only |
| genuine mean | 0.682079 (TR centred 512) | 0.6839 (512-SIE), median 0.7057 | no |

The closest recorded-to-measured pair is the centred-512-with-camera row: FNMR 48.6626 % vs 48.1540 % (Δ 0.51 pp), genuine mean 0.6821 vs 0.6839 (Δ 0.0018), EER 1.678 % @ 0.330 vs 1.5242 % @ 0.32755. Those are **near-agreements in sign and magnitude**, which is informative, but they are not reproductions in the CA-01 sense because the recorded side declares no pool. **Verdict: proposal §6 and report §4.4 are consistent with the artifacts, and their refusal to difference against `verification_threshold_audit.json` is required, not stylistic.**

Population sensitivity is documented on disk as swinging FNMR 0.021 % → 18.70 % → 93.98 % for the same operating point purely by pair rule (`ES-EXP07-AUDIT-FORENSIC-01.md` B-11, `JETSON-RUN-SHEET-PREP-01-REV01.md:129`); the artifacts behind 0.021 % and 93.98 % are **not on disk** (`UNVERIFIED`, though the swing direction is confirmed by this pass: changing only the camera rule moves FNMR by 12.58 pp — 60.7384 % ↔ 48.1540 % between the agnostic and SIE pools of the same arm's scores). The proposal's requirement that "any FNMR quoted without its population is invalid" is satisfied by every FNMR in REV02/report REV02 and violated by every recorded one.

### CA-13 — The recorded production recommendation contradicts the recorded device status

`QUOTED` — `gate3_real_results_reconciled.json` in one file contains `"jetson_orin_nano_status": "NOT MEASURED — Laptop timing rejected for edge deployment decisions"` **and** `"recommendation": "Deploy OSNet x1.0 under TensorRT FP16 on Jetson Orin Nano. Reject TransReID due to 40x memory overhead and inferior edge efficiency."` The charter's hypothesis to be tested ("…might break the critical TTO handoff budget") is thus already answered on disk by the quantity that file says was never measured. Also in the same file: `"root_cause": "Genuine gold-standard baseline verified against official Torchreid model zoo"` for the OSNet row, on a host where that extractor cannot be imported (CA-16) — the claim is about a different environment and is `UNVERIFIED` here. Proposal REV02 §8 and report REV02 §7 both explicitly refuse this pattern; **the refusal is consistent, the record is not.**

---

## 7. Assets, provenance, and the OSNet baseline

### CA-14 — Projection assets are not name-unique, and the recorded variance fingerprints which file each run used

`COMPUTED` (this pass) + prior audit. `models/transreid/` and `models/projections/` hold **8 `.npz` with 4 distinct contents**: `9626bea5…` ×2 (`_real_v2`, `_real_v3_reproduce`, 1,466,587 B, R₅₁₂ 0.9864118), `dc2b9072…` ×3 (`svd_projection_768_to_512.npz`, `…_real.npz`, `projections/…`, 1,466,710 B, R₅₁₂ 0.9857139), `c9bca759…` ×2 (`_norm05_centered`, 1,464,564 B), `628b1dc6…` ×1 (`_norm05`, non-centred, mean all zeros). The suffix "real" therefore does **not** distinguish a file — selecting by name can silently pick a different matrix, and `edgesync_bringup/config/algorithm_thresholds.yaml:47` selects the projection **by name** (`reid_projection_path: "models/transreid/svd_projection_768_to_512_real_v2.npz"`), so a rename or replacement silently changes the embedding space.

The fingerprint that matters for this audit: `gate3_real_results.json` records `svd_variance_retained = 0.9857138991355896` (= the `dc2b9072…` family), while `_v2_corrected` and `_v3_reproduce` record `0.9864118099212646` (= the deployed `9626bea5…` family). **So v1 was computed with a different projection asset, not only different preprocessing** — which strengthens CA-03: the recorded 71.17 → 93.38 "correction" moved at least three things (projection file, camera/SIE, and normalisation) with only one named as the cause. `COMPUTED` from the two `svd_variance_retained` values and the measured R₅₁₂ of each family.

### CA-15 — Training-exposure evidence is consistent with the proposal's claim

`COMPUTED`. Both checkpoints carry classifier heads over **751** classes (`TransReID` 5 × (751,768); `OSNet` (751,512)+bias), matching the measured `bounding_box_train` identity count of 751 and *not* the test count of 750 (+ junk `−1`). `gate3_real_results_reconciled.json` records `n_training_identities: 751` — agreement. `TransReID`'s `sie_embed` is (6,1,768) = six cameras ✓. Proposal §5's split-by-folder rule and §W-02's "no test leakage" claim are therefore corroborated by asset structure, not only by folder arithmetic. **Consistent, and stated as corroboration rather than proof.**

### CA-16 — The OSNet baseline cannot run here; three distinct failure layers

`COMPUTED` (this pass and change-log §5 rows 17/25), inspection only per §3B:

1. `osnet_trt` path — `models/osnet_trt_extractor.py:72-75` returns `np.zeros(512)` and never touches an execution context (proposal P1). No engine exists to execute.
2. `osnet_torch` path — `models/osnet_torch_extractor.py:64` `from torchreid.models import build_model` raises **`RuntimeError: operator torchvision::nms does not exist`** on this host; the fallback at `:74` is `except ImportError`, so the measured `RuntimeError` **is not caught** and propagates out of `extract_embedding`, which violates the documented contract in `interfaces/base_identity_extractor.py` (must return the `zeros(512)` sentinel, must not raise). `COMPUTED` from the recorded failure string and the source lines.
3. Even when the fallback is entered directly, `_build_osnet_standalone()` (`:93-124`) is a placeholder — measured 8 entries / 42,945 elements (42,816 trainable + 129 buffers) versus the checkpoint's 2,554,771 trainable elements; **0 of 567** key names shared; `strict=False` load reports `missing_keys = 7 / unexpected_keys = 567`. It consumes none of `osnet_x1_0_market1501.pth`, and its `preprocess()` (`:140-143`) uses ImageNet stats. The embeddings manifest records the attempt as `FAILED — recorded, not repaired (§3B)` with `effect_on_comparison` stated.

**Consequence for the proposal:** every recorded OSNet accuracy number (CA-01 table, `94.32897862232778 / 97.80285035629454 / 98.60451306413302 / 83.5692348350484`, byte-identical across four files plus the forensic table) came from an environment that is not this one, and no per-query outcomes exist. `NOT MEASURED` here; the paired test is `NOT APPLICABLE`; §3B forbids repair. This is also the single largest threat to any recommendation (report REV02 §6 threat 11).

### CA-17 — `run_real_pipeline.py`, the only producer script on disk, points at an absent tree

`COMPUTED`. `scripts/run_real_pipeline.py:33` hard-codes `BRINGUP = r"c:\Users\PC\OneDrive\Pictures\EdgeSync-main (3)\EdgeSync-main\Implementation Phase\edgesync_ws\src\edgesync_bringup"`; that directory **does not exist on this host** (`test -d` → ABSENT). Its `:40` output path is `gate3_real_results_v2_corrected.json`, which does exist in the project copy. **So the artifact that the current population-provenance note cites as its source (`evaluate_offline.py:80` → `run_real_pipeline.py:488-498` → `gate3_real_results_v2_corrected.json`) cannot be regenerated from this tree without editing that constant** — report-only, not repaired (it is neither a TransReID blocker nor §3C-eligible). The reproduction this audit relies on is instead the one this pass ran through `scripts/generate_task7_embeddings.py` + `analyze_task7_statistics.py`, which resolve paths from CLI arguments (change-log §5 rows 17/22/23, exit 0).

---

## 8. Proposal text vs what exists

| Proposal claim (REV01, as written) | On-disk state, `COMPUTED` | Consistent? |
|---|---|---|
| "`tensorrt` imported by 2 files, declared in no `package.xml`, **no `.engine`/`.plan`/`.onnx`/`.trt` file exists**" | `.engine/.plan/.trt`: still **none**. `.onnx`: **two now exist** (agnostic 343,380,183 B `318b6857585d5d7c`, SIE 343,399,114 B `2cd8ccc774c59337`, checker PASS, opset 17) | **Partly superseded** — REV02 §1 updates this row. "TensorRT 8.6" remains `UNVERIFIED`: `tensorrt` is not importable on this host and no version was read from a device |
| "shared `TrtEngineRunner` runtime" | Class **does not exist**; the TransReID TRT extractor is self-contained and unit-tested with stubbed `tensorrt`/`pycuda` only | REV02 §3 records as-designed vs as-built; **no engine inference was run** |
| Selection via `reid_backbone` | Parameter does not exist; the live key is `reid_model_type` (`identity_node.py:57-60`, `algorithm_thresholds.yaml:34`, `extractor_factory.py:38-89`) | REV02 corrects it. **`ablation_task7_osnet.yaml`, `ablation_task7_transreid.yaml`, `task7_identity_ablation.launch.py`: not created** |
| "`config/` installed, `models/` and `data/` not" (`edgesync_bringup/setup.py:11-16`) | unchanged | Consistent; still blocks an installed-space engine path |
| §6 "50 warm-up / 200 timed / 3 trials" vs v01's "1,000 calls" | v01's 1,000-call figure has no artifact; every recorded latency block uses 200 or records none | REV02 §6 fixes it. The harness exists (`benchmark_transreid_jetson.py`, protocol block written by its own refusal path) but **executed zero calls** — change-log §5 row 19, exit 1 by design, `BLOCKED — Pending Jetson Orin Nano access`, zero latency numbers recorded |
| §5 "the SVD projection must be re-fitted and re-versioned" (v01) / "never re-fit here" (REV02) | Deployed projection **not re-fitted**: hash-pinned `9626bea5…` in the manifest and re-verified by the consumer's integrity gate | Consistent with REV02 and with charter §5's prohibition |
| G.1–G.7 owner decisions | **G.5 answered** (per-robot `reid_camera_id` — implemented and both arms measured). G.1/G.3/G.7 open or denied | Both REV02 documents mark H7b/H7c `UNVERIFIED — NOT MEASURED` for exactly this reason. Consistent |
| "REV01 supersedes `ES-EXP07-TASK7-PROPOSAL-01.md` (sha256-16 `aa9f5f7dbf9d9127`), preserved byte-for-byte in `es07_audit/backup/`" | `es07_audit/` **absent**; `ES-EXP07-TASK7-PROPOSAL-01.md` and `ES-EXP07-TASK7-REPORT-01.md` **absent** from the tree | **NOT RECONCILABLE** — the v01 documents and their asserted digests cannot be verified. REV02's header states this. The only lineage that *is* verifiable on disk is REV01 → REV02 (`845aeb70c5fdbd0f` / `f5951ac3511f9e39`, re-measured unchanged after the REV02 edits) |
| Prior audit's own inventory | `docs/…/artifact-inventory.md` prefixes match my recomputed digests for `forensic_ablation_results.json` (`a283f66d554c9b94`), `_reconciled` (`564323622381700a`), `_v3_reproduce` (`915f55547649ed20`); `docs/…/audit_20261019_141628/model_asset_inventory.md:85-86` quotes **different** full digests (`1fdfe660…`, `ca7657ea…`) for the same two files | **The team's two inventory documents disagree with each other**; this pass agrees with `artifact-inventory.md`. Anyone quoting a size/digest should use the re-measured values in §2 of this file |

---

## 9. Defects found in this pass's own tooling (disclosed, not hidden)

1. **CA-02's own error.** The `id_band` assumption was repeated in REV02's first draft and in the statistics consumer's `not_measured` reason string before the artifacts were read. The consumer's string remains on disk uncorrected (rationale in CA-02); the documents are corrected.
2. **ONNX parity coverage.** `transreid_onnx_report.json` records `parity_n 8` for the exported graphs, while the *embedding* run's parity cross-check ran on **every** image (`every_nth 1`, PASS). The graph-level export parity claim is therefore on 8 images; the extractor-level claim is on 19,281. Both are stated in the report; neither is presented as more than it measured.
3. **Manifest latency is not a benchmark** and was initially a candidate for §4.1; excluded, and the exclusion is written into report §8 (CA-07 item 4).
4. **MiB/MB conversions** in the first REV02 draft were wrong by 0.02–0.04 and the node counts were asserted as 783 for both graphs; re-measured to 327.47 / 327.49 MiB and 779 (agnostic) / 783 (SIE) nodes, 154 / 155 initializers. Corrected before publication.
5. **Two earlier EER implementations** (a 53.5 M-distinct-score enumeration, then a bounded bracket search) — the first exceeded its timeout and produced **no artifact**; the replacement is what generated CA-01's table, and the suite carries a cost guard so that shape fails a test instead of a run (change-log §5, after row 25).

---

## 10. What may and may not be quoted from this experiment

**May be quoted, with its label:**
- `transreid_512_sie` Rank-1 93.3789 % / Rank-5 97.9513 % / Rank-10 98.8717 % / mAP 85.1252 % on 3,368 × 15,913 — `COMPUTED` **and** `REPRODUCED` (CA-01).
- The within-TransReID SIE effect: Rank-1 **+4.6318 pp**, mAP **+7.8012 pp**, per-query genuine-accept **+7.60 pp**, false-accept Δ −0.24 pp (Holm p = 4.361e-41 / 2.427e-72 / 0.5728) — `COMPUTED`, declared family of 3, declared pair pool 49,649 genuine / 53,532,220 impostor / 13,115 duplicates excluded.
- The calibration finding: every arm's EER (1.52–2.01 %) lies at τ ≈ 0.306–0.367, TAR 0.90 needs τ ≈ 0.455–0.508, and the SIE arm's genuine **mean** 0.6839 is *below* the shipped τ while its **median** 0.7057 is barely above it — `COMPUTED`. τ was not changed; re-calibration is an owner decision.
- Exact arithmetic of the four recorded model-size figures and the three legitimate size ratios (CA-05, CA-06).
- Every defect in §3–§8 above, with file, line and digest.

**May not be quoted:**
- Any TransReID-vs-OSNet conclusion, in either direction. There is no runnable OSNet arm here (CA-16) and no per-query baseline outcomes exist (CA-01). The recorded 94.329/83.569 row is `QUOTED`, not verified.
- Any latency, throughput, engine size, resident-memory Δ, peak-memory or FP16 figure as an **edge** number — `NOT MEASURED`, and the offline set contradicts itself in sign (CA-07) with the smaller model appearing to use more memory (CA-12).
- Any FNMR/FMR/EER difference against `verification_threshold_audit.json` (CA-10) or against the forensic 768-D rows (CA-11).
- The reconciled file's recommendation, in either direction (CA-13).
- `es07_audit/backup/` or the v01 documents as evidence — they are not on disk.

**Verdict.** The proposal and the artifacts are **consistent on accuracy, on τ provenance, on asset pinning and on the split-by-folder rule**, once REV02's corrections are applied. They are **inconsistent on latency, on every verification-pool number, on the memory-footprint method, on the "different population" explanation, on the 768-D arm values, and on the recorded production recommendation** — and five of the seven recorded result files cannot be regenerated from this tree. The hypothesis under test (H7) remains **undecided on this host** for two of its three axes and **unverified-by-measurement** for the third, because the baseline arm cannot run; nothing in the recorded evidence supports the charter's stated expectation that TransReID's accuracy rise is paid for by edge latency, and nothing refutes it either. **Do not let any downstream document quote "40.5×", "200+ ms", or "OSNet is optimal" without naming which definition it used and which device measured it — after this audit, those three phrases are the specific points where the proposal and the results on disk do not agree.**
