# ES-EXP07 Task 7 — Implementation Report (TransReID re-identification integration)

Deliverable 1 of the ES-EXP07 Master Completion & Verification Prompt §8. Scope is **Experiment 7 /
Task 7 (TransReID) only**. Companion documents: `ES-EXP07-TASK7-INTEGRATION-VERIFICATION-REPORT.md`
(commands, exit codes, counts, §7 checklist), `ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` (every
SHA-256 before/after and the protected-asset table), and `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md`
(the running execution record this report summarises rather than replaces).

Authority for what counts as "done": this prompt's §4.1–§4.7 requirements, plus the earlier ES-EXP07
charter items TR-01…TR-11 that Task 7 was already carrying. Both sets are listed below, each with the
issue, the fix, the files, and the class of evidence that supports it.

## 1. Ground rules that shaped every implementation choice

| Rule (§ of the prompt) | How it constrained the code |
|---|---|
| §1.1 Task 13 / CBBA / coordination off-limits | No negotiation protocol, coordination algorithm or ownership-FSM file was implemented, repaired or investigated. The only coordination touch is §4.7's removal of byte-compiled leftovers (§6) |
| §1.2 No Git writes; filesystem evidence only | All change detection is SHA-256 + byte size + directory walk. `git` was never invoked (the tree is not a repository: `git status` → exit 128) |
| §1.3 Canonical `reid_model_type`, alias `reid_model`, deterministic precedence, conflicts loud | Implemented twice — at the config resolver and at the node — because either layer can be the first to see a conflicting pair (§4.3) |
| §1.4 No full re-extraction; no overwrite of recorded outputs; add `--exp-id` | The §4.4 offline work was proven on a **2-image** run into a scratch directory; the 19,281-image artefacts were re-hashed, not rewritten |
| §1.5 Evidence classes | Every claim below carries STATIC / UNIT / INTEGRATION / HARDWARE. No static or mocked result is described as ROS 2 or hardware verification |
| §4.6 Protected assets, protocol unchanged | No checkpoint, projection, normalisation, architecture, threshold or pair rule was altered; the six named assets are byte-identical |
| §9 Don't stop for approval on ordinary reversible Task 7 changes | 12 source/config/test files were written under the prompt's standing authorisation — 7 modified and 5 created (1 config + 4 test modules) — each modification preceded by a verified backup |

## 2. Task 7 issues inherited from the preflight audit (TR-01…TR-11)

| ID | Issue (as verified, not assumed) | Fix implemented | Files | Evidence class / status |
|---|---|---|---|---|
| TR-01 | `reid_model_type` was a `DeclareLaunchArgument` with `default_value='osnet_mock'`, and the four `reid_*` values were injected as a dict **after** `algo_params` — in ROS 2 the last source wins, so the launch defaults silently overwrote the YAML and the deployed arm could never be TransReID | Pure resolver module `reid_launch_defaults.py` reads the YAML(s) in Python and derives the launch defaults from what was actually configured; the launch file calls it once, so one channel decides precedence | `edgesync_bringup/edgesync_bringup/reid_launch_defaults.py`, `edgesync_bringup/launch/robot_bringup.launch.py`, `tests/test_reid_launch_defaults.py` | UNIT → `PASS`; live launch check `INTEGRATION PENDING` |
| TR-02 | Failed inference returned `np.zeros(512)` on six paths, and zeros flowed into the verifier as similarity 0.0 → a fabricated "MISMATCH"; the offline evaluator injected zeros for undecodable images | Three boundaries: the node **refuses** instead of scoring (invalid candidate/reference detected before the verifier, message says `INVALID_INPUT`/`INVALID_EMBEDDING` and the reason is logged); the extractor keeps the sentinel (the base class is a frozen contract that *requires* it) but makes it attributable via `last_failure` + `failure_counts` + ERROR log; the evaluator returns `kept_indices`/`excluded` instead of zeros rows | `identity_node.py`, `models/transreid_extractor.py`, `core/reid_input_adapter.py`, `scripts/evaluate_offline.py`, 3 test modules | UNIT → `PASS` at all three boundaries; frozen contract `base_identity_extractor.py` untouched and re-hashed identical |
| TR-03 | The deployed arm could not receive a camera label, so SIE was unreachable in production even though the recorded arm used it; `IdentityVerify.srv` has no camera field and is frozen for this task | Per-robot integer parameter `reid_camera_id`, mapped 1-based → 0-based by `resolve_camera_label()` (`<=0` camera-agnostic, out-of-range falls back to agnostic with the reason logged), passed to `extract_embedding()` only when the extractor accepts it. Service contract unchanged, byte-identical | `identity_node.py`, `tests/test_identity_node_transreid_integration.py` | UNIT → `PASS`; live SIE `INTEGRATION`/`HARDWARE PENDING`. §4.2.8 close-out additionally proved the value arrives as an `int`, not a string (§4.2 below) |
| TR-04 | The ROS image boundary ignored `msg.encoding`, so BGR arrived at a model trained on RGB | Explicit adapter at the boundary: encoding-driven channel order, padded-row stride handling, refusal before inference for an unsupported encoding, and a round-trip check against the offline evaluator's `cv2.cvtColor(BGR2RGB)` convention so production and evaluation feed the extractor the same pixels | `core/reid_input_adapter.py`, `tests/test_transreid_input_adapter.py` (31 cases), node-side integration file | UNIT → `PASS`; a real `sensor_msgs/Image` cannot be exercised here (`rclpy` absent) |
| TR-05 | Asset paths resolved against the process CWD, so the same config worked or failed depending on where `ros2 launch` was invoked | Resolution through the package share directory with absolute paths returned unchanged; asset existence validated before construction | `identity_node.py`, `extractor_factory.py` (`_require_file`), tests | UNIT → `PASS`; installed-share resolution `INTEGRATION PENDING` |
| TR-06 | `load_state_dict()`'s result was discarded, so a partially-loaded or schema-mismatched checkpoint looked like success | Coverage validation: missing keys must be empty, unexpected keys must equal the tolerated set, parameter count asserted; on the real checkpoint `missing_keys == []` and `torch_parameters == 100,769,512` | `models/transreid_extractor.py`, `tests/test_transreid_extractor_guardrails.py` | UNIT + real-checkpoint load → `PASS` (heavy cases are the two opt-in skips in the current run) |
| TR-07 | The evaluation population was an invalid ID band, so "the published test set" was not what was measured | Named, CLI-selected populations with per-keeps/exclusion-reason metadata, `RuntimeError` on an empty split, and `NOT APPLICABLE` (not `0.0`) where a population contains no distractors | `scripts/evaluate_offline.py`, `tests/test_evaluate_offline_population.py` (28 cases) | UNIT → `PASS`; the published-population measurement it enabled is the 19,281-embedding run (change log §5 rows 17–23) |
| TR-08 | Verification metrics lacked pair metadata; EER/TAR solvers produced phantom values on empty pools | `analyze_task7_statistics.py`: SHA-256 integrity gate on the inputs, τ read from the shipped config rather than remembered, Wilson intervals, bootstrap CIs, McNemar + Holm paired tests only where both arms have valid per-query predictions, FNMR/FMR/EER with pair counts, and COMPUTED / REPRODUCED / BLOCKED / NOT MEASURED / NOT APPLICABLE labels | `scripts/analyze_task7_statistics.py`, `tests/test_analyze_task7_statistics.py` (68 cases) | UNIT + INTEGRATION for the script's own `main()` → `PASS`. A performance regression found in its first EER rewrite killed one run and is disclosed in the change log §5 |
| TR-09 | No ONNX artifact existed, so deployment claims had no graph behind them | Exporter with graph validation, dynamic-batch checks, input-contract refusal and PyTorch↔ONNX parity on real crops; byte-identical re-export proven | `scripts/export_transreid_onnx.py`, `tests/test_export_transreid_onnx.py` (33 cases) | real inference + real export → `PASS`. **The graphs live outside the repo**; `find . -name '*.onnx'` inside the tree is 0 |
| TR-10 / TR-10b | No TensorRT arm in the production dispatch | `transreid_trt_extractor.py` + registry entry `transreid_trt`, numpy preprocessing bit-identical to the Torch arm's (heavy parity test), fail-closed when no engine/`tensorrt` is present | `models/transreid_trt_extractor.py`, `models/extractor_factory.py`, 2 test modules | UNIT (stubbed TRT) → `PASS`; execution `HARDWARE PENDING`, and no engine is claimed to exist |
| TR-11 | Jetson benchmarking had no harness, and a laptop number risked being reported as a device number | Build + benchmark harnesses that record the target environment and **refuse** to run on a non-target host; `--allow-non-target` labels its output `NOT A DEPLOYMENT ARTEFACT` | `scripts/build_transreid_trt_engine.py`, `scripts/benchmark_transreid_jetson.py` | measured refusal (exit 1 + BLOCKED reports) → `HARDWARE PENDING` |

## 3. The §4.1–§4.7 integration work (this prompt's own requirements)

### §4.1 Registry-based factory — `PASS` (UNIT)

`extractor_factory.py` now dispatches through `BACKBONE_REGISTRY` (`:137`) of `BackboneSpec` records
(`:47`) holding id, implementation, description and a builder. Five ids are registered: `osnet_mock`,
`osnet_torch`, `osnet_trt`, `transreid`, `transreid_trt` (`:32-36`). Contract validation runs **before**
construction (`_resolve_registered_class` `:219`), and the returned instance is checked against the
registry by class **identity**, not `isinstance` (`_verify_against_registry` `:224`), so a look-alike
subclass cannot impersonate an arm. Builders resolve their classes lazily through `importlib`, which is
what lets a host with no TensorRT still import the factory — pinned by a test that spawns a minimal
interpreter and asserts the import succeeds and the TRT module never enters `sys.modules`.

### §4.2 `exp7_transreid.yaml` really loaded — `PASS` (UNIT), live launch `PENDING`

One channel decides what the node receives:
`robot_bringup.launch.py:38` calls `select_experiment_config(algo_params)` → `:39`
`resolve_reid_config(algo_params, experiment_config)`, which reads the shared
`algorithm_thresholds.yaml` **first** and the experiment overlay **second**, so only keys the overlay
defines override. The resolved values become the `DeclareLaunchArgument` defaults and the injected
parameter dictionary; an explicit `reid_model_type:=…` on the command line still wins.

Measured behaviours (change log §5 row 33, 25/25): the default path is unchanged when no overlay is
selected and the entity list stays flat (8 arguments + 7 nodes, so the shipped launch tests remain
valid); a missing *optional* overlay preserves every shared value; a missing **explicit** overlay raises
`FileNotFoundError` naming the path; a malformed overlay raises `ValueError` carrying the YAML parse
context; an overlay selecting an unregistered model id raises instead of degrading; package-relative and
config-dir-relative overlay paths both resolve, preferring an existing file.

**§4.2.8 integer camera id.** `reid_camera_id` is declared as an integer parameter
(`identity_node.py:92`, `declare_parameter('reid_camera_id', 0)`). A launch *string* cannot be delivered
to an integer parameter, so the launch file coerces at parse time (`coerce_camera_id` at
`reid_launch_defaults.py:128`, applied at `robot_bringup.launch.py:80`, injected at `:136`) and the
overlay writes `0` as a YAML integer. Verified in both directions: the injected object is an `int` and
**not** a `LaunchConfiguration`, and an overlay value of `3` reaches the node as int `3`.

**Deliberate deviation, documented.** §4.2.9 asked for "an appropriate launch argument or equivalent
mechanism". A launch argument cannot select the overlay, because `generate_launch_description()` builds
every default before the CLI is parsed — an `experiment_config:=…` value would either be invisible to the
resolver or arrive through a second parameter-file channel whose precedence necessarily conflicts with the
injected dictionary that carries the resolved values. The equivalent mechanism chosen is the
`EDGESYNC_EXPERIMENT_CONFIG` environment selector (`reid_launch_defaults.py:80`), which keeps one channel
and one precedence order:

```bash
EDGESYNC_EXPERIMENT_CONFIG="<abs>/src/edgesync_bringup/config/exp7_transreid.yaml" \
  ros2 launch edgesync_bringup robot_bringup.launch.py
```

The overlay's header states this in-file, including that the `experiment_config:=…` form is **not** wired,
so a reader cannot be misled by a command that looks supported.

### §4.3 Canonical + alias with loud conflicts — `PASS` (UNIT)

`reid_model_type` stays canonical; `reid_model` is accepted as a backward-compatible alias. Precedence is
explicit-canonical → explicit-alias → established shared default, implemented at **both** layers, because
the config resolver and the node can each be the first to see the pair. When two explicit values disagree
the canonical value is kept **and** a warning names both keys and both resolved ids — never a silent pick;
agreement is case- and whitespace-insensitive; an unknown id raises listing every registered value.
Measured in rows 31 (config layer, `describe_reid_sources` origin labels such as
`model_id='transreid' source=canonical:experiment`) and 32 (node layer, `model_source=explicit:reid_model`
for the alias-only case).

### §4.4 `exp_id` chain, runtime and offline — `PASS` (UNIT + real CLI)

Runtime: `exp7_transreid.yaml` (`exp_id: "EXP07"`) → `reid_launch_defaults.DEFAULT_EXP_ID` (`:77`) →
`DeclareLaunchArgument('exp_id')` (`robot_bringup.launch.py:65-66`) → injected into the node's parameter
dictionary (`:135`) → `identity_node.py:89` `declare_parameter('exp_id', 'EXP07')` → logged at `:202` as
`EXP-METADATA exp_id=EXP07 reid_model=transreid model_source=explicit:reid_model_type robot_id=ugv_2
camera_id=0`. No `.msg`/`.srv` file changed. `exp_id` is a string parameter and is delivered as a
`LaunchConfiguration`; the camera id is the integer case above; other nodes received no Re-ID keys.

Offline (§1.4's explicit requirement, and the part that had been accepted-then-discarded before this
pass): `generate_task7_embeddings.py` takes `--exp-id` (default `EXP07`, `:270`), refuses an empty value
(`:278`), logs `EXP-METADATA exp_id=… arms=… output_dir=…` (`:293`) and stamps `manifest['exp_id']`
(`:338`) while leaving the pre-existing `experiment` label text unchanged. `analyze_task7_statistics.py`
takes `--exp-id` (same default constant, `:93`, `:1090`) plus `--allow-exp-id-mismatch`, and its
`resolve_exp_id()` gate (`:713`) runs **immediately after the manifest is loaded and before any metric is
computed** (`:1109`), with four measured outcomes:

| Situation | Outcome recorded |
|---|---|
| manifest id == requested id | `agreement: match`, `action: proceed`; report carries the label and the markdown header states it |
| manifest id ≠ requested id | `action: abort` — `SystemExit` naming both labels and both remedies, **no JSON written** |
| mismatch with `--allow-exp-id-mismatch` | `agreement: mismatch`, `action: proceed_with_override`, the disagreement kept in the report as a visible note |
| manifest predates §4.4 (no `exp_id`) | `agreement: manifest_has_no_exp_id`, `action: proceed` with an explicit WARNING that the numbers are reported under the requested label "on that assumption alone" |

An `ast`-based test pins three things a future edit could silently break: the two scripts' `DEFAULT_EXP_ID`
constants equal each other, each CLI's `default=` is that same `Name` node (not a literal), and the bringup
default plus the node's `declare_parameter('exp_id', 'EXP07')` agree with them.

### §4.5 No silent `osnet_mock` — `PASS` (UNIT)

Three independent closures: (a) an `ImportError` raised by a builder is converted to a `RuntimeError`
whose message states it "will NOT fall back to 'osnet_mock'" (`extractor_factory.py:280`), so a missing
TensorRT install fails the TRT arm rather than quietly returning a pseudo-embedding; (b) the registry's
class-**identity** check rejects wrong concrete classes and look-alike subclasses, so a mock cannot
masquerade as a real arm; (c) at the config layer an explicitly requested `osnet_mock` is rendered as a
`NOTE:` naming the selection, while an accidental or absent selection produces a `WARNING:` with the
origin — the two cases are distinguishable in the launch output. Real TRT selection on this host (no
engine, no `tensorrt`) raises, which is asserted by a test that also proves the absence of TensorRT on
the host.

### §4.6 / §1.4 Protected assets and recorded outputs untouched — `PASS` (measured)

See manifest §C (6/6 byte-identical, sizes included), §D (full-tree re-hash: 244 identical / 6 changed /
0 missing / 2 hash-skipped-by-design, `osnet_in_changed_or_missing: []`, 10 OSNet entries identical), and
§F (recorded embedding manifests and statistics outputs re-hashed identical after the pass). The
scientific protocol was not touched: no projection refit, no checkpoint change, no normalisation,
architecture, τ or pair-rule edit. τ remains read-from-config at 0.70.

### §4.7 Authorised removals — `PASS`

`edgesync_coordination/…/negotiation/` + its `__pycache__` and 5 `.pytest_cache` directories removed after
their contents were walked and recorded; 0 `.py` sources were present, all three `.pyc` preserved outside
the repo with digests, and the four *other* `__pycache__` directories in that package were deliberately
left alone because §4.7 does not authorise removing generated artifacts by name. Post-removal state was
re-measured, not inferred from the exit code (manifest §E).

## 4. Defects found during this pass

**Fixed (in scope, Task 7 files only):**
1. `analyze_task7_statistics.markdown()` crashed with `TypeError` on a zero-valid-query arm while still
   labelling it `COMPUTED` at `0.00 %` and writing `NaN` coverage means. Now `—` intervals, a
   `NOT MEASURED — … no denominator` label and `null` means; 3 regression cases added. No previously
   reported number changes, because no recorded run has a zero-valid population.
2. `test_extractor_factory_transreid_trt.py`'s lazy-import test asserted a source string that the §4.1
   registry refactor made unreachable. Re-pointed to a behavioural check in a fresh interpreter (stronger,
   still runs without TensorRT); no assertion was weakened or removed.
3. Three of my own test fixtures in `test_exp7_config_overlay.py` were wrong, not the production code: two
   asserted "no warning" against a missing / Re-ID-key-free shared file, which *legitimately* warns. The
   fixtures now give both layers real files that load cleanly. Two `match=` regexes in
   `test_extractor_factory_registry.py` did not match the production sentences (`not` vs `NOT`); pinned to
   the exact text instead of loosening the pattern.
4. `run_mini_pytest.py` (scratch tooling, outside the repo) mis-reported a genuinely-absent module as the
   workspace `launch` shadow. Fixed so the runner names the module found in the exception text first.

**Reported, not modified (out of scope per §1.1/§1.2 and the frozen-OSNet rule):**
- `config/algorithm_thresholds.yaml`'s root key `algorithm_thresholds` matches no node name in
  `robot_bringup.launch.py`, so it is not consumed as a ROS 2 parameter file anywhere in bringup. The
  resolver reads it through Python; the project-wide namespace defect is left alone per §4.2's "do not
  broadly refactor every bringup YAML".
- `signature_smoother.py`'s `_zero_streak` reference in `identity_node.py` silently yields `-1`
  (observability only).
- OSNet arm: `osnet_torch_extractor.py` cannot import (`torchreid` → `torchvision::nms does not exist`),
  and the team's `TestOSNetStandalone*` tests pass against a 42,816-parameter placeholder that shares 0 of
  the checkpoint's 567 keys — green tests there do not evidence a working OSNet arm. Recorded and
  quantified in the consistency audit; not repaired.
- Five of seven `results/experiment7/*.json` have no producer on disk, and the recorded latency and 768-arm
  numbers disagree between artifacts. Documented in `TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`.

## 5. What is **not** implemented, stated plainly

- No ROS 2 runtime verification of any kind: no node actually launched, no parameter actually read from a
  live node, no `EXP-METADATA` line observed in a real ROS log, no service round-trip. All of that is
  `INTEGRATION PENDING`, with the exact commands listed in the verification report §7.
- No TensorRT engine, no device execution, no Jetson latency/power/memory numbers. `HARDWARE PENDING`.
  Nothing in this pass moves a laptop measurement into a device slot.
- No change to the message/service contracts, the OSNet baseline, the projection, the checkpoint, or the
  evaluation protocol; no full-population re-extraction.
- Experiment selection for **other** experiments (Task 13 and beyond) was not generalised: the overlay
  mechanism is generic in code but only `exp7_transreid.yaml` is wired, because §1.1 forbids reaching into
  other experiments' configuration on this task's authority.
