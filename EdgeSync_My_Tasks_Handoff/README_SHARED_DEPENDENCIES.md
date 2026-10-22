# Shared dependencies

Files that Task 7 or Task 13 needs in order to import, configure or run, but which are **not**
this task's implementation work. Nothing in this folder was edited by the packaging run.

| | Rows |
|---|---:|
| SHARED project files (`SHARED_PROJECT_DEPENDENCY`) | 65 |
| `BASELINE_PROTECTED` (read-only by charter) | 51 |
| `OTHER_TEAM_OWNED` | 3 |
| Physically delivered under `source_snapshot/` | 47 |

## 1. What is here and why

| Group | Examples (original paths under `source_snapshot/`) | Why it is included |
|---|---|---|
| Message / service definitions | `edgesync_msgs/msg/*.msg` (12), `edgesync_msgs/srv/IdentityVerify.srv` | Both tasks exchange these; the ReidSignature contract is the Exp-7 → Exp-13 interface |
| Build metadata | `edgesync_msgs/CMakeLists.txt`, `package.xml` files, `setup.py`, `package/` markers | A reviewer must be able to see how the workspace builds and where the messages are declared |
| Baseline coordination | `edgesync_coordination/edgesync_coordination/nodes/coordination_node.py`, `ownership_fsm.py`, `epoch_manager.py`, `utility_engine.py`, `eligibility_gate.py`, `negotiation_manager.py`, `liveness_watchdog.py` | Task 13 extends these; the Exp-13 node inherits them, so their exact bytes are needed to read the diff |
| Upstream producers | `edgesync_perception/.../perception_node.py`, `edgesync_acquisition/.../target_acquisition_node.py` | They are the intended upstream of the identity chain (see the integration map) and the evidence for the missing crop publisher |
| Bring-up | `edgesync_bringup/launch/robot_bringup.launch.py`, `config/*.yaml`, `fastdds_profiles.xml` | The configuration surface the two experiments are launched into |

## 2. Ownership discipline (brief §6)

* **51 `BASELINE_PROTECTED` rows** — OSNet backbone/extractors, baseline FSM/epoch/utility
  machinery and the shared message set. Charter §3B freezes them; they were read, hashed and
  copied, never modified.
* **3 `OTHER_TEAM_OWNED` rows** — the perception and acquisition nodes.
* **Verified unchanged**: of the 54 protected/other-team rows, all
  54 that also exist in the 2026-10-16 archive are
  **byte-identical to it**, and their current digests equal the digests recorded in the out-of-repo
  pre-change backups (65 packaged files have a matched earlier version).

## 3. Two shared files that differ from the 2026-10-16 archive

These are **pre-existing project state**, changed by the earlier authorised implementation sessions
and documented in the change manifests — not by this packaging run:

| File | Current digest | Earliest backup digest | Recorded in |
|---|---|---|---|
| `robot_bringup.launch.py` | `45550fc4b3ed4084…` | `c3ddc747f7be2727…` (edgesync_es07_backup/modified_20261019_exec) | `TASK07/manifests/ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` |
| `algorithm_thresholds.yaml` | `3576df5a194e491b…` | `5879b513fb43f686…` (edgesync_es07_backup/modified_20261019_exec) | `TASK07/manifests/ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` |

## 4. Reference copies, not a second master

To keep each task slice independently executable, 62
dependency rows are mirrored inside `TASK07/source_snapshot/` and `TASK13/source_snapshot/`. Every
mirror is recorded as an extra entry in the same inventory row's `copies[]` list, is flagged
`reference_copy: true`, and says in its `purpose` that the authoritative copy is the same path in
the project tree. `source_snapshot/` in this folder is the authoritative copy within the package.

## 5. Not included

* No `__pycache__`, `.pyc`, `.pytest_cache`, `build/`, `dist/` or virtual-environment content.
* No unrelated experiments, and no other teammate's whole package merely because one of my modules
  imports it — only the specific files on the two tasks' import path.
* No dataset, checkpoint or synthetic fixture above the size policy; those are listed with digests
  in `../README.md` §5.