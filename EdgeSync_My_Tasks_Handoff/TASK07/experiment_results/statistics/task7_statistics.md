# Task 7 statistics — measured from `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory`

Generated 2026-10-19T16:06:52Z on {'platform': 'win32', 'python': '3.10.11', 'numpy': '1.26.4'}. Every figure is labelled COMPUTED / REPRODUCED / BLOCKED / NOT MEASURED / NOT APPLICABLE; a cell that is not COMPUTED says why.

## Retrieval (population: `directory`)

| Arm | dim | Queries valid | Rank-1 % (95 % CI) | Rank-5 % | Rank-10 % | mAP % (bootstrap CI) | Label |
|---|---|---|---|---|---|---|---|
| `transreid_768_agnostic` | 768 | 3368 / 3368 | 88.75 (87.64–89.77) | 96.02 | 97.92 | 77.33 (76.49–78.20) | COMPUTED |
| `transreid_512_agnostic` | 512 | 3368 / 3368 | 88.75 (87.64–89.77) | 96.14 | 97.86 | 77.32 (76.49–78.18) | COMPUTED |
| `transreid_512_sie` | 512 | 3368 / 3368 | 93.38 (92.49–94.17) | 97.95 | 98.87 | 85.13 (84.42–85.85) | COMPUTED |

## Verification at the shipped threshold

τ = 0.7 read from `algorithm_thresholds.yaml` (`algorithm_thresholds.ros__parameters.reid_similarity_threshold`); equals 0.70: **True**. Accept iff similarity >= τ, the production rule. Appearance gate only — the Mahalanobis spatial gate is NOT MEASURED offline (no trajectory data exists).

| Arm | genuine pairs | impostor pairs | FNMR at τ (95 % CI) | FMR at τ (95 % CI) | EER | τ at EER | FMR at TAR=0.90 |
|---|---|---|---|---|---|---|---|
| `transreid_768_agnostic` | 49,649 | 53,532,220 | 54.66 % (54.23–55.10) | 0.0079 % (0.01–0.01) | 2.01 % | 0.3671 | 0.2975 % |
| `transreid_512_agnostic` | 49,649 | 53,532,220 | 60.74 % (60.31–61.17) | 0.0052 % (0.01–0.01) | 1.97 % | 0.3058 | 0.2810 % |
| `transreid_512_sie` | 49,649 | 53,532,220 | 48.15 % (47.71–48.59) | 0.0058 % (0.01–0.01) | 1.52 % | 0.3276 | 0.1377 % |

## Paired inference (declared family, Holm-corrected)

- **H1** `rank1_hit` on transreid_512_agnostic vs transreid_512_sie: rates 0.8875 vs 0.9338 (Δ -0.0463), discordant 5 / 161 of 3368 pairs, exact McNemar p = 2.18039e-41, Holm-adjusted 4.36078e-41 → **REJECT H0 at alpha=0.05**
- **H2** `genuine_accept_at_tau` on transreid_512_agnostic vs transreid_512_sie: rates 0.8109 vs 0.8869 (Δ -0.0760), discordant 3 / 259 of 3368 pairs, exact McNemar p = 8.09013e-73, Holm-adjusted 2.42704e-72 → **REJECT H0 at alpha=0.05**
- **H3** `false_accept_at_tau` on transreid_512_agnostic vs transreid_512_sie: rates 0.1992 vs 0.2016 (Δ -0.0024), discordant 73 / 81 of 3368 pairs, exact McNemar p = 0.572846, Holm-adjusted 0.572846 → **FAIL TO REJECT H0 at alpha=0.05**

- OSNet vs TransReID paired test: **NOT APPLICABLE on this host** — FAILED — recorded, not repaired (charter §3B): RuntimeError: operator torchvision::nms does not exist

## Camera-partition proxy

A camera partition on one fixed dataset. It measures whether appearance identity survives a viewpoint change between two camera groups — the property a handoff depends on — but involves no robot, no handoff event, no trajectory and no handoff label. Charter §8 explicitly forbids inferring cross-robot handoff success or false-handoff rates from Market-1501 retrieval metrics, so nothing in this section is a handoff measurement.

- `transreid_768_agnostic` **cams_[1, 2, 3]_queries_to_cams_[4, 5, 6]_gallery**: 1843 valid queries of 1915 (gallery rows 7752), Rank-1 90.02 % (88.56–91.30), Rank-10 97.99 %, mAP 81.17 %; FNMR at τ 54.20 %, FMR at τ 0.0069 %
- `transreid_768_agnostic` **cams_[4, 5, 6]_queries_to_cams_[1, 2, 3]_gallery**: 1449 valid queries of 1453 (gallery rows 8161), Rank-1 91.10 % (89.52–92.46), Rank-10 98.27 %, mAP 83.03 %; FNMR at τ 53.38 %, FMR at τ 0.0065 %
- `transreid_512_agnostic` **cams_[1, 2, 3]_queries_to_cams_[4, 5, 6]_gallery**: 1843 valid queries of 1915 (gallery rows 7752), Rank-1 89.91 % (88.45–91.20), Rank-10 97.99 %, mAP 81.28 %; FNMR at τ 60.28 %, FMR at τ 0.0046 %
- `transreid_512_agnostic` **cams_[4, 5, 6]_queries_to_cams_[1, 2, 3]_gallery**: 1449 valid queries of 1453 (gallery rows 8161), Rank-1 91.30 % (89.74–92.65), Rank-10 98.14 %, mAP 82.78 %; FNMR at τ 59.12 %, FMR at τ 0.0043 %
- `transreid_512_sie` **cams_[1, 2, 3]_queries_to_cams_[4, 5, 6]_gallery**: 1843 valid queries of 1915 (gallery rows 7752), Rank-1 93.76 % (92.56–94.78), Rank-10 98.97 %, mAP 87.74 %; FNMR at τ 48.22 %, FMR at τ 0.0056 %
- `transreid_512_sie` **cams_[4, 5, 6]_queries_to_cams_[1, 2, 3]_gallery**: 1449 valid queries of 1453 (gallery rows 8161), Rank-1 94.20 % (92.88–95.29), Rank-10 98.76 %, mAP 87.60 %; FNMR at τ 48.04 %, FMR at τ 0.0051 %

## What is not measured here

- **jetson_tensorrt_latency**: BLOCKED — Pending Jetson Orin Nano access (TR-11)
- **osnet_retrieval_on_this_population**: BLOCKED on this host — the frozen baseline cannot be imported (torchvision::nms); §3B forbids repairing it
- **spatial_gate_mahalanobis**: NOT MEASURED — requires per-robot trajectory and covariance data that do not exist offline
- **cross_robot_handoff_success**: NOT MEASURED and NOT INFERABLE — no robot or handoff dataset exists in this project
- **statistical_comparison_against_recorded_gate3_numbers**: NOT APPLICABLE — the recorded run used the invalid id_band population (TR-07), so its per-query outcomes are not the same population as this one
