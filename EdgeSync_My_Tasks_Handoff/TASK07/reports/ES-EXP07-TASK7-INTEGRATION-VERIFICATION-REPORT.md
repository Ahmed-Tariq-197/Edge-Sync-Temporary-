# ES-EXP07 Task 7 — Final Verification Report (integration close-out, §4.1–§4.7)

Deliverable 2 of the ES-EXP07 Master Completion & Verification Prompt §8. Every number here is a
transcription of a captured log; nothing is a remembered or estimated figure. File-level before/after
evidence is in `ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md`; the work-item mapping is in
`ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §2 (CLOSE-01…07) and §5 rows 30–43.

## 0. Verdict

**`SOFTWARE-INTEGRATION COMPLETE`** under this prompt's §7 vocabulary — every *feasible local software*
criterion in the 17-item checklist passes with a recorded outcome, and the two external categories
(ROS 2 runtime, Jetson/TensorRT hardware) are marked pending, not passed.

**`PARTIALLY COMPLETE`** remains the correct verdict for ES-EXP07 **as a project**, because the standing
charter treats Jetson Orin Nano execution as a required target activity rather than a documentation note,
and that has not happened. The two statements are not in conflict: §7 of this prompt scopes its status
wording to software integration, and this report does not use that scoping to imply hardware coverage.

No claim of `FULL HARDWARE VERIFIED` is made or implied anywhere in this pass.

## 1. Environment actually present (measured, not assumed)

| Probe | Result |
|---|---|
| OS / interpreter | Windows 10.0.26300 x64 · Python 3.10.11 · torch 2.5.1+cu121 · cuDNN 90100 · RTX 3050 Laptop 4.00 GiB |
| `which ros2` | **absent**, and invoking it → **exit 127** `ros2: command not found` |
| `import rclpy` / `launch_ros` / `ament_index_python` | **exit 1** — `ModuleNotFoundError` for each |
| `import tensorrt` / `import pycuda` | **exit 1** — absent (the TRT arm therefore must and does fail closed) |
| `importlib.util.find_spec('launch')` from the repo root | **None** |
| `import launch` with cwd = `src\edgesync_bringup` | **exit 0** — *not* ROS 2: the workspace `launch\` directory shadows the absent package as a namespace dir. This is the exact mechanism behind the shipped launch test's `ImportError: cannot import name 'LaunchDescription' from 'launch' (unknown location)` |
| `colcon --help` | exit 0 (a Python 3.10 `colcon` shim is installed). **`colcon build`/`colcon test` were not attempted**: with no `ament_index_python`/`launch_ros` they would write `build/`, `install/` and `log/` trees into the workspace while starting no middleware, i.e. zero verification value for a real artifact cost |
| pytest | installed, **not authorised on this host in this session** — see §3 |

## 2. §5.1 required validation steps, each mapped to measured output

| §5.1 step | What was run | Exit | Outcome | Class |
|---|---|---|---|---|
| 1. Syntax/compile check of every modified file | `python -m py_compile` ×11 (the 7 modified Python files of manifest §A + the 4 new test modules) | **0** | `PY_COMPILE_EXIT=0` for all 11 | STATIC |
| 2. Config parse checks | `yaml.safe_load` of `exp7_transreid.yaml` and `algorithm_thresholds.yaml` | **0** | `YAML_OK …exp7_transreid.yaml ['/**']`; `YAML_OK …algorithm_thresholds.yaml ['algorithm_thresholds']` — the overlay has the single root key a ROS 2 parameter file needs | STATIC |
| 3. Focused tests for the changed behaviour | `run_mini_pytest.py … test_exp7_offline_exp_id.py test_analyze_task7_statistics.py` | **0** | **collected 92 · passed 92 · skipped 0 · failed 0** (24 new §4.4 cases + the 68-case shipped analyzer suite as the regression gate) | UNIT |
| 4. Full feasible Task 7 suite — identity package | 16 targeted modules (change log §5 row 30) | **0** | **collected 375 · passed 373 · skipped 2 · failed 0**. The 2 skips are the deliberate `ES07_HEAVY` opt-ins (real-checkpoint guardrail, TRT↔Torch preprocess parity) | UNIT |
| 5. Full feasible Task 7 suite — bringup package | `test_exp7_config_overlay`, `test_reid_launch_defaults`, `test_robot_bringup_launch` | **0** | **collected 57 · passed 57 · skipped 0 · failed 0**; the third module is unimportable (row 43's shadow mechanism) and is counted as *not run*, not as a pass | UNIT |
| 6. Node-layer verification harness | `verify_node_identity_resolution.py` | **0** | **20/20 checks passed** — §4.3 precedence incl. conflict WARN, §4.4 `EXP-METADATA` line, §4.2.8 integer camera id, unknown id → `ValueError` listing registered ids | UNIT (stub `rclpy`) |
| 7. Launch-description harness | `verify_launch_description.py` | **0** | **25/25 checks passed** — flat 8-arg/7-node entity list preserved, overlay precedence, `reid_camera_id` injected as a real `int`, `exp_id` as a `LaunchConfiguration`, missing → `FileNotFoundError`, malformed → `ValueError`, unknown id → raise, env selector `('', 'none')` vs path | UNIT |
| 8. CLI flags in a real subprocess with synthetic fixtures | §3 of manifest row 34–36 + change log rows 34, 35, 36 | 0 / 1 / 0 as listed | `--exp-id` in `--help`; `--exp-id ''` → exit 2 `cannot be empty`; real 2-image extraction stamps `manifest['exp_id'] = 'EXP07-verify'`; handoff match / abort / override measured on that real artefact | UNIT + real-CLI |
| 9. Existing CLI invocations still compatible | change log row 34 (shipped 68-case analyzer suite unchanged and green) and row 30 (the team's `test_experiment7_fixes`, `test_experiment7_forensic_fixes`, `test_identity_fusion`, `test_gate1_scaffolding` all in-process green) | **0** | no pre-existing test changed state; the 0.70 threshold config path, `--arms`, `--population`, `--skip-camera-partition` all still work | UNIT |
| 10. Inspect **all** failures, not only the first | every non-zero exit this pass was read and attributed | — | 4 distinct failure signatures, each investigated to a cause: `test_identity_service.py` (no `rclpy`), `test_robot_bringup_launch.py` (`launch` namespace shadow), the shipped lazy-import test (obsolete source-string assertion → §5 below), and the markdown writer crash on a zero-valid arm (real defect → §6 below) | STATIC + UNIT |

## 3. The pytest disclosure, stated first because it changes how to read every count above

`pytest` is installed here, and rows 1–29 of the change log §5 report pytest counts (152 → 235 → 303 →
305) from earlier passes where it **was** run under an earlier authorisation. **In this pass it was not
authorised, so it was not run, and no count in this report is a pytest count.** Rows 30, 31 and 34 used
`C:\Users\PC\edgesync_es07_runs\task7close_verify\run_mini_pytest.py` — a direct-interpreter runner whose
own docstring limits it to the features these suites use (module functions, `test_*` classes,
`setup_method`, `@pytest.mark.parametrize`, the `tmp_path`/`monkeypatch` built-ins, module-level fixtures,
a `logs` handler capture) and states that anything it cannot emulate is reported **FAIL, never silenced or
counted as a pass**. Consequences the reader should hold:

- Coverage per file and per assertion is the same files and the same asserts, but parametrisation, fixture
  scoping and collection are the runner's, not pytest's. Treat counts as **UNIT on this host**, nothing more.
- The two historical "305/305" and "303/305" figures are **not current evidence** (§3 Phase A instruction).
  The current, repeatable local figure is **375 / 373 / 2 / 0** (identity) and **57 / 57** (bringup).
- Restoring real pytest evidence needs one line of authorisation; the commands are in §7 below.

## 4. Requirement-by-requirement outcome for §4.2–§4.6

| Requirement | Where it is enforced in code | Measured proof | Verdict |
|---|---|---|---|
| §4.2 shared first, experiment second; only overlay keys override | `reid_launch_defaults.py:264` `resolve_reid_config()` | row 33: overlay changes `exp_id`/`reid_camera_id` while leaving the shared checkpoint at its shared value | PASS (UNIT) |
| §4.2 missing *optional* overlay preserves shared; missing *explicit* overlay fails | `:229` `select_experiment_config()` + `:159` `_read_parameters(strict)` | row 33: no env var → `('', 'none')` and the default path is byte-for-byte the pre-overlay entity list; explicit missing file → `FileNotFoundError` naming the path | PASS (UNIT) |
| §4.2 malformed overlay must fail clearly, not silently degrade | `:159` catches parse errors and re-raises `ValueError` with context | row 33: `ValueError: [reid_launch_defaults] malformed YAML in …: while parsing a flow sequence…` | PASS (UNIT) |
| §4.2 a mechanism selects the experiment configuration | `:80` `EXPERIMENT_CONFIG_ENV_VAR = 'EDGESYNC_EXPERIMENT_CONFIG'`, called once at `robot_bringup.launch.py:38` | row 33 + row 31 (`test_exp7_config_overlay.py`) | PASS (UNIT). Design note: a **launch argument** was tried and rejected — `generate_launch_description()` builds every `DeclareLaunchArgument` default before the CLI is parsed, so `experiment_config:=…` would either be invisible to the resolver or arrive through a second precedence channel. The env-var selector keeps one channel and leaves the shipped flat entity list intact, which is why the team's launch tests still pass. This is a *deliberate deviation from §4.2.9's suggested wording*, documented rather than hidden |
| §4.2.8 `reid_camera_id` reaches the node as a valid **integer** | `coerce_camera_id()` at `reid_launch_defaults.py:128`, applied at `robot_bringup.launch.py:80`, injected at `:136`; declared `declare_parameter('reid_camera_id', 0)` at `identity_node.py:92` | row 33 asserts `isinstance(injected, int)` and rejects a `LaunchConfiguration`; overlay value 3 arrives as int 3; row 32 asserts the node reads 2 as an int | PASS (UNIT) |
| §4.2.10 keep the established Python-read-and-inject path; no broad YAML refactor | parameters are passed as `[shared_yaml, resolved_dict]` with the dict last | row 33: identity params are exactly `[yaml, dict]`, 2 entries, order preserved; `algorithm_thresholds.yaml` untouched at `3576df5a194e491b` | PASS |
| §4.3 canonical / alias / default precedence, conflicts never silent | node `identity_node.py:213-235`; config layer `reid_launch_defaults.py:70,264+` | row 32 (6 precedence cases incl. conflict-keeps-canonical + WARN naming both keys, case/whitespace-insensitive agreement); row 31 (config-layer equivalent) | PASS (UNIT) |
| §4.4 `exp7_transreid.yaml → reid_launch_defaults.py → robot_bringup.launch.py → identity_node.py → logs`, default `EXP07`, no msg/srv change | overlay `exp_id: "EXP07"`; `:77`; launch `:65-66,135`; node `:89`, log `:202` | rows 32, 33 for the runtime chain; row 30/34 for the node and offline tests; `IdentityVerify.srv` re-hashed identical (`1eae43d251dd1cde`) in the manifest §C/§D | PASS (UNIT). Real ROS 2 log line = INTEGRATION PENDING |
| §4.4 offline `--exp-id` must not be accepted and then discarded | generator stamps `manifest['exp_id']` (`generate_task7_embeddings.py:65,270,338`) + `EXP-METADATA` log `:293`; analyzer gate `analyze_task7_statistics.py:93,713,1090,1109` | **rows 35–36 on a real artefact**: stamped, read back, match logged, mismatch aborts before computing, override records `proceed_with_override`, legacy manifest reports `manifest_has_no_exp_id`; `report['exp_id']` and the markdown header both carry the label | PASS (UNIT + real CLI) |
| §4.5 no silent `osnet_mock` | registry `extractor_factory.py:137,219-245`; `ImportError` → `RuntimeError` `:280` "…will NOT fall back to 'osnet_mock'"; class **identity** check in `_verify_against_registry` `:224` | row 30 (`test_extractor_factory_registry.py`: wrong concrete class → `TypeError`, look-alike subclass rejected, `transreid_trt` with no engine raises instead of returning a pseudo-embedding, and the factory imports in a minimal interpreter with no optional deps); row 32 unknown-id `ValueError` listing all 5 registered ids; row 31 explicit `osnet_mock` surfaces as `NOTE:`, not silently | PASS (UNIT) |
| §4.6 protected assets byte-identical | no edit attempted | manifest §C: **6/6 `byte_identical = true`** with sizes; §D full-tree re-hash `osnet_in_changed_or_missing: []`, 10 OSNet-path entries identical | PASS (measured) |

## 5. Defect in a shipped test, re-pointed rather than deleted

`test_extractor_factory_transreid_trt.py::…lazy import…` asserted
`source.count('from ..models.transreid_trt_extractor import') == 1` against
`inspect.getsource(create_extractor)`. §4.1 moved dispatch into `BACKBONE_REGISTRY`, whose builders resolve
classes through `importlib`, so that literal no longer exists anywhere in the module — the assertion could
only ever fail from now on, and it was failing. The test's *intent* (the TRT module must not enter
`sys.modules` unless the TRT arm is requested) is now asserted **in a fresh interpreter**: build
`osnet_mock`, then require that none of `tensorrt`, `pycuda`,
`edgesync_identity.models.transreid_trt_extractor` is in `sys.modules`, and that construction succeeded.
That is a stronger check than a string count and it runs on a host with no TensorRT (§4 row 46; the file
went `508896cc7b20d80a` → `f189401b1e16dc85`, +25/−6 lines). No assertion was weakened to reach green, and
nothing was skipped or deleted to get there.

## 6. Real defect found by measurement in `analyze_task7_statistics.py` (fixed)

Running the §4.4 override path on a 2-query smoke artefact produced an arm with **zero valid queries** and
`markdown()` died with `TypeError: 'NoneType' object is not subscriptable`, while the same row was still
labelled `COMPUTED` at `0.00 %` and `identity_coverage` means were `NaN`. Three separate honesty problems in
one place: a crash, a fabricated-looking zero, and `NaN` reaching JSON.

Fix (Task 7 script only, no metric definition or pair rule touched): retrieval rows use the existing
`_ci_text()` helper so an absent interval renders `—`; a zero-valid arm is labelled
`NOT MEASURED — none of the N queries has a gallery identity match, so Rank-k and mAP have no denominator`
instead of `COMPUTED`; coverage means become `None`, never `NaN`. Re-measured on the same artefact →
**exit 0**, row renders `0.00 (—) | 0.00 | 0.00 | 0.00 (—)` with the `NOT MEASURED` label and `null` means.
Pinned by 3 regression cases (`TestDegenerateSmokePopulation`, inside row 34's 92). **No recorded run has a
zero-valid population, so no previously reported Task 7 number changes.**

## 7. §5.2 / §5.3 — the two phases that cannot be executed here, and their exact commands

`rclpy`, `launch_ros`, `ament_index_python` and `ros2` are absent, so §5.2 was **not performed and is not
claimed**. Per §5.2's instruction the commands are recorded for the host that can run them, in order:

```bash
# 0. build (from the workspace root, ROS 2 environment sourced)
cd "Implementation Phase/edgesync_ws" && colcon build --symlink-install

# 1. default bringup — prove the path is unchanged when no overlay is selected
ros2 launch edgesync_bringup robot_bringup.launch.py
#    expect: identity_node starts, EXP-METADATA exp_id=EXP07 reid_model=transreid
#            model_source=… camera_id=0, no osnet_mock anywhere

# 2. overlay selected — prove exp7_transreid.yaml is actually loaded
EDGESYNC_EXPERIMENT_CONFIG="$PWD/src/edgesync_bringup/config/exp7_transreid.yaml" \
  ros2 launch edgesync_bringup robot_bringup.launch.py

# 3. parameter truth at runtime (the §4.2.8 type check, on a real node)
ros2 param get /identity_node exp_id          # -> string 'EXP07' (or the overlay value)
ros2 param get /identity_node reid_camera_id  # -> integer, NOT a string
ros2 param get /identity_node reid_model_type # -> 'transreid'

# 4. alias + conflict behaviour on a live node (§4.3)
ros2 launch edgesync_bringup robot_bringup.launch.py reid_model:=transreid
ros2 launch edgesync_bringup robot_bringup.launch.py reid_model_type:=transreid reid_model:=osnet_torch
#    expect: canonical wins AND a WARN naming both keys — never a silent pick

# 5. the team's launch-level test, unrunnable here (§5 row 31/43)
cd src/edgesync_bringup && python -m pytest tests/ -q --continue-on-collection-errors

# 6. service round-trip that needs real middleware (test_identity_service.py, §5 row 30)
cd src/edgesync_identity && python -m pytest tests/test_identity_service.py -q
```

Restoring **local pytest** evidence (authorisation only, nothing else missing) is one line:

```bash
cd "Implementation Phase/edgesync_ws/src/edgesync_identity" && \
  python -m pytest tests/ --ignore=tests/test_identity_service.py -p no:cacheprovider -q
# and: cd ../edgesync_bringup && python -m pytest tests/ -p no:cacheprovider -q --continue-on-collection-errors
```

**§5.3 Jetson / TensorRT — `HARDWARE PENDING`, prerequisites not present on this host:** an Orin Nano with
L4T (`/etc/nv_tegra_release`), `tensorrt`, `pycuda`, `trtexec`, `nvpmodel`, `jetson_clocks`, `tegrastats`,
the exported ONNX graphs (which exist, outside the repo), and the two harnesses already in the tree
(`scripts/build_transreid_trt_engine.py`, `scripts/benchmark_transreid_jetson.py`). Both harnesses were
executed on this non-target host and **refused with exit 1 and a written BLOCKED report** (change log §5
rows 18–19), which is the only hardware-related evidence this pass can offer: proof that the tooling cannot
mistake a laptop for the target, not a deployment result. The §7 run sheet
`JETSON-RUN-SHEET-PREP-01-REV01.md` holds the on-device sequence.

## 8. §7 acceptance checklist — 17 items, with the honest residual on each

| # | Criterion | Verdict | Evidence / residual |
|---|---|---|---|
| 1 | Existing baseline integrity confirmed | **MET** | manifest §D: 252 entries, 244 identical, 6 changed (all owned by this task), 0 missing; §A 21/28 unchanged |
| 2 | `identity_node.py` syntax and import verified | **MET to the feasible local limit** | `py_compile` exit 0 (row 39); the module imports and runs in the stub-`rclpy` harnesses (rows 30, 32). A real `import rclpy` needs ROS 2 → §7 commands |
| 3 | Registry-based factory implemented and tested | **MET** | §4 row 40; rows 30/32/34 |
| 4 | Existing model IDs and constructors remain compatible | **MET** | 5 registered ids incl. the three OSNet arms unchanged (`osnet_torch`/`osnet_trt` digests identical, manifest §A); rows 30/32 |
| 5 | `exp7_transreid.yaml` exists and is actually loaded through the intended path | **MET (UNIT)** | rows 31, 33; the loading code is the only channel and its output was inspected, not assumed. Runtime proof on a live node = §7 command 2 |
| 6 | Shared/experiment precedence deterministic | **MET** | row 33 including the "overlay leaves shared keys it does not define" case |
| 7 | `reid_model_type` and `reid_model` alias both work with tested precedence | **MET** | rows 31 (config layer) + 32 (node layer), conflict path emits WARN and keeps canonical |
| 8 | `exp_id=EXP07` propagates through runtime and logs | **MET (UNIT)** | rows 32 (`EXP-METADATA exp_id=EXP07 reid_model=transreid model_source=explicit:reid_model_type robot_id=ugv_2 camera_id=0`), 33 (launch default + injected `LaunchConfiguration`), and `test_identity_node_model_resolution.py` at the node layer. Real log line on a live node = §7 commands 1–3 |
| 9 | Offline scripts support `--exp-id`, default `EXP07`, record it in output metadata | **MET** | rows 34–36 on a real artefact, both default and custom values; §4.4 default agreement between the two scripts is pinned by an `ast` check, so it cannot silently drift |
| 10 | Offline changes validated with synthetic/small fixtures only | **MET** | row 34 (synthetic two-arm fixture) + row 35 (**2 images**, `--limit 2`) |
| 11 | Accidental `osnet_mock` fallback is explicit and cannot masquerade as a real model | **MET** | §4.5 row of table §4; class-identity check rejects look-alikes; `osnet_mock` appears only on explicit request and renders as `NOTE:` |
| 12 | Protected assets remain byte-identical | **MET** | manifest §C 6/6 + §D OSNet entries |
| 13 | No full dataset extraction was rerun | **MET** | row 38; recorded manifests re-hashed identical |
| 14 | All feasible Task 7 tests have actual recorded outcomes | **MET, with the pytest caveat in §3** | rows 30–34 logs kept outside the repo with sizes and digests |
| 15 | No unrelated Task 13 implementation or coordination changes | **MET** | `edgesync_coordination` Python sources unchanged; the only coordination-side touch is §4.7's removal of 3 `.pyc` + 5 `.pytest_cache`, with `negotiation_dir_py_sources: []` recorded first |
| 16 | SHA-256 change manifest and final execution report exist | **MET by this pass** | `ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` + this report + the implementation report + handoff checklist |
| 17 | Environment-dependent ROS 2 / Jetson items clearly marked pending, not falsely passed | **MET** | §7 above; rows 43; handoff checklist columns "verified with ROS 2" and "verified on Jetson/TensorRT" are **empty by measurement**, not by omission |

## 9. Limitations (what this report does not establish)

1. **No ROS 2 runtime was ever started.** Everything labelled UNIT ran in-process against real files and
   real launch functions, or as real CLI subprocesses of the offline scripts. No message was published, no
   service called, no node discovered. `test_identity_service.py` and `test_robot_bringup_launch.py` did not
   run and are **INTEGRATION PENDING**.
2. **No hardware evidence.** No TensorRT engine, no Jetson, no device latency, no power or memory figures.
   **HARDWARE PENDING**. Row 35's 2-image pass has a `latency_mean_ms` field; on a 4 GB laptop GPU over 2
   images it is a pipeline artifact, not a benchmark, and is not quoted as one.
3. **The runner is not pytest** (§3). Counts are correct for these files and assertions; they are not a
   reproduction of pytest's collection or fixture semantics.
4. **The heavy parity checks were not re-run.** `ES07_HEAVY=1` loads the 415 MB checkpoint for the guardrail
   and the TRT↔Torch preprocess parity; those two cases are the only 2 skips in row 30. Earlier passes ran
   them (rows 3, 10) and they are historical, not this pass's evidence.
5. **Overlay selection deviates from the prompt's suggested mechanism** (env-var selector instead of a
   launch argument) for the measured reason in §4. If a launch argument is required for Gate-review
   symmetry, the fix is a second resolver pass *after* CLI parsing, not a second parameter channel; that is
   an owner decision, not something to change silently.
6. **Pre-existing out-of-scope defects remain**: `algorithm_thresholds.yaml`'s root key matches no node name
   (so it is consumed through Python, not as a parameter file); `signature_smoother.py`'s `_zero_streak`
   `-1` sentinel; the OSNet arm's `torchvision::nms` import failure and the green-but-placeholder OSNet
   standalone-backbone tests (§3B: recorded, not repaired).

## 10. Artifacts written by this pass (all outside the repository)

`C:\Users\PC\edgesync_es07_runs\task7close_verify\` — `run_mini_pytest.py`,
`verify_node_identity_resolution.py`, `verify_launch_description.py`, and the captured logs with their
digests: `FINAL_identity.log` 55,670 B `ee35948b14ba34ca` · `FINAL_bringup.log` 5,170 B
`59d11feedb9d3399` (also captured earlier as `bringup_overlay_verify.log`, same digest) ·
`FINAL_node_resolution.log` 1,707 B `9d3e7f838a9b830d` · `FINAL_launch_description.log` 8,434 B
`79ec5af82db84bf3` · `exp44_offline_verify.log` 9,659 B `b38d797faf3c1ac5` · `expid_smoke_run.log`
1,959 B `d45a0c527348688d` · `exp44_offline_measured.txt` (command-by-command record) ·
`FINAL_change_inventory.json` · `FINAL_protected_assets.json` · `expid_smoke\` (2-image embeddings,
manifest stamped `EXP07-verify`, and the three analysis output dirs).
`C:\Users\PC\edgesync_es07_runs\scope_rehash_task7close.py` →
`scope_rehash_after_task7close_20261019.json`.
Backups: `C:\Users\PC\edgesync_es07_backup\pre_task7close_20261019\` (+ `BASELINE_MANIFEST.json`),
`pre_exp44_offlineid_20261019\`, `preserved_20261019_task7close\`.
