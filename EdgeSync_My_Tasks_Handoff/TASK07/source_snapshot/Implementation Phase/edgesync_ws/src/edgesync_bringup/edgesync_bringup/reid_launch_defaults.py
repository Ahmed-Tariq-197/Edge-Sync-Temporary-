"""Re-ID / experiment configuration resolution for bringup (ES-EXP07 Task 7, TR-01 + §4.2-§4.5).

Why this module reads YAML itself
---------------------------------
``robot_bringup.launch.py`` passes its Re-ID keys to ``identity_node`` as a parameter dictionary
placed *after* ``algorithm_thresholds.yaml`` in the ``parameters=[...]`` list, so the dictionary
wins. With literal launch defaults the configured model selection and asset paths were
unreachable: bringup always selected the mock extractor and never passed a checkpoint (TR-01).

Reading the YAML here lets each ``DeclareLaunchArgument`` default *be* the configured value, so an
explicit ``reid_model_type:=...`` still overrides it.

Experiment overlay (§4.2)
-------------------------
``resolve_reid_launch_defaults(shared, experiment)`` layers the shared config first and the
experiment config second; a key the experiment file does not define keeps its shared value.
Precedence of an explicitly requested experiment file is strict:

* shared file missing or malformed  -> documented silent fallback to ``HARDCODED_FALLBACKS``, with
  the reason recorded in the provenance so ``describe_reid_sources`` can print it;
* experiment file not requested (empty path) -> shared behaviour, unchanged;
* experiment file requested but missing or malformed -> ``FileNotFoundError`` / ``ValueError``.
  Bringup must never report that ``EXP07`` loaded when it did not.

``select_experiment_config`` is the one selector for which experiment overlay to apply. A launch
argument cannot be it: ``generate_launch_description()`` builds every ``DeclareLaunchArgument``
default before the CLI is parsed, so a launch-argument overlay path would either be invisible to
this resolver or would fight the injected parameter dictionary for precedence. The environment
variable ``EDGESYNC_EXPERIMENT_CONFIG`` is read instead, and the resolver stays the single channel
that decides what the node receives.

Model identifier alias (§1.3, §4.3)
-----------------------------------
``reid_model_type`` is canonical; ``reid_model`` is a backward-compatible alias accepted from
config files. Deterministic precedence, highest first:

1. ``reid_model_type`` from the experiment file
2. ``reid_model_type`` from the shared file
3. ``reid_model`` from the experiment file
4. ``reid_model`` from the shared file
5. ``HARDCODED_FALLBACKS['reid_model_type']`` — recorded as an *implicit* origin (see §4.5)

If a source supplies both keys with different values, the canonical key wins and the conflict is
reported rather than silently dropped. The resolved value is what this module returns for
``reid_model_type``, so the launch dictionary always carries the canonical selection.

Typing (§4.2.8)
---------------
``reid_camera_id`` is an INTEGER parameter on ``identity_node``. A launch substitution can only
produce a string, so this module never places ``reid_camera_id`` into a launch parameter
dictionary: ``coerce_camera_id`` validates the configured value here, and the typed value reaches
the node through the ROS 2 parameter file (``/**: ros__parameters:``), which rclpy parses with the
correct type. Overriding it on the command line must therefore use
``--ros-args -p reid_camera_id:=3`` against the node, not a launch substitution.
"""
from typing import Any, Dict, List, Optional, Tuple
import os

#: Canonical ROS parameter names consumed by ``identity_node`` for Re-ID.
REID_LAUNCH_KEYS = (
    'reid_model_type',
    'reid_checkpoint_path',
    'reid_projection_path',
    'reid_engine_path',
    'reid_camera_id',
    'exp_id',
)

#: Guide-name alias accepted in config files, resolved onto the canonical key.
MODEL_ALIAS_KEY = 'reid_model'

#: Every key this module may return. ``reid_model`` is reported so the node can also detect a
#: conflicting alias, but the authoritative value is always ``reid_model_type``.
RESOLVED_KEYS = REID_LAUNCH_KEYS + (MODEL_ALIAS_KEY,)

#: Experiment identification default (Task 7). Override per experiment in its own YAML.
DEFAULT_EXP_ID = 'EXP07'

#: The only per-run selector for the experiment overlay; see ``select_experiment_config``.
EXPERIMENT_CONFIG_ENV_VAR = 'EDGESYNC_EXPERIMENT_CONFIG'

#: Used when a config is absent or carries none of the keys. Identical to the literal defaults
#: this launch file used before TR-01, so a missing config degrades to the previous behavior.
HARDCODED_FALLBACKS: Dict[str, str] = {
    'reid_model_type': 'osnet_mock',
    'reid_checkpoint_path': '',
    'reid_projection_path': '',
    'reid_engine_path': '',
    'reid_camera_id': '0',
    'exp_id': DEFAULT_EXP_ID,
    MODEL_ALIAS_KEY: '',
}

#: Static mirror of the registered backbone identifiers, used when ``edgesync_identity`` is not
#: importable at launch time. Kept in sync by tests/test_exp7_config_overlay.py.
STATIC_BACKBONE_IDS = ('osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt')

#: Provenance labels reported by describe_reid_sources().
STATUS_LOADED = 'loaded'
STATUS_ABSENT = 'absent'
STATUS_MALFORMED = 'malformed'
STATUS_NO_REID_KEYS = 'no_reid_keys'


def known_backbone_ids() -> Tuple[str, ...]:
    """Return the registered backbone ids, or the static mirror if the package is unavailable.

    The authoritative source is ``extractor_factory.BACKBONE_REGISTRY``. Importing it is optional
    so that bringup never fails because the identity package is not on ``PYTHONPATH``.
    """
    try:
        from edgesync_identity.models.extractor_factory import registered_backbone_ids
        return tuple(registered_backbone_ids())
    except Exception:
        return STATIC_BACKBONE_IDS


def normalize_model_id(value: str) -> str:
    """Return the registry-normalized form of a configured model identifier."""
    return str(value or '').strip().lower()


def is_valid_model_id(value: str) -> bool:
    """True when ``value`` names a backbone this project can construct."""
    return normalize_model_id(value) in known_backbone_ids()


def coerce_camera_id(value: Any) -> int:
    """Validate a configured ``reid_camera_id`` as an integer.

    ``0`` keeps inference camera-agnostic; ``1..6`` selects the SIE embedding. A non-integer
    raises because a string reaching an INTEGER ROS parameter fails at runtime, far from here.
    """
    text = str(value).strip()
    try:
        camera_id = int(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"[reid_launch_defaults] reid_camera_id must be an integer, got {value!r}. "
            f"Set it as a typed value under '/**: ros__parameters:' or pass "
            f"'--ros-args -p reid_camera_id:=<int>' to identity_node."
        ) from None
    if camera_id < 0:
        raise ValueError(
            f"[reid_launch_defaults] reid_camera_id must be >= 0 (0 = camera-agnostic), "
            f"got {camera_id}."
        )
    return camera_id


def read_config_status(yaml_path: str) -> str:
    """Classify a configuration file without raising: loaded / absent / malformed / no_reid_keys."""
    found, _error, status = _read_parameters(yaml_path, strict=False)
    if status != 'ok':
        return STATUS_MALFORMED if status in ('malformed', 'not_a_mapping') else STATUS_ABSENT
    return STATUS_LOADED if (set(found) & set(RESOLVED_KEYS)) else STATUS_NO_REID_KEYS


def _read_parameters(yaml_path: str, strict: bool):
    """Return ``(found, error, status)`` for one file.

    ``status`` is ``'ok'`` or the reason it is not (absent / unreadable / malformed /
    not_a_mapping). With ``strict`` a failure raises instead: an explicitly requested experiment
    file must never be silently discarded.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML ships with ROS 2
        if strict:
            raise RuntimeError(f"[reid_launch_defaults] PyYAML unavailable: {exc}") from exc
        return {}, f'pyyaml_unavailable: {exc}', 'unreadable'

    if not yaml_path or not os.path.isfile(yaml_path):
        message = f'file not found: {yaml_path}'
        if strict:
            raise FileNotFoundError(
                f"[reid_launch_defaults] requested configuration is missing — {message}. "
                f"Refusing to continue as if the experiment had loaded."
            )
        return {}, message, 'absent'

    try:
        with open(yaml_path, 'r', encoding='utf-8') as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        if strict:
            raise FileNotFoundError(
                f"[reid_launch_defaults] cannot read {yaml_path}: {exc}"
            ) from exc
        return {}, str(exc), 'unreadable'
    except yaml.YAMLError as exc:
        message = f'malformed YAML in {yaml_path}: {exc}'
        if strict:
            raise ValueError(f"[reid_launch_defaults] {message}") from exc
        return {}, message, 'malformed'

    if not isinstance(loaded, dict):
        message = f'{yaml_path} is not a YAML mapping'
        if strict:
            raise ValueError(f"[reid_launch_defaults] {message}")
        return {}, message, 'not_a_mapping'

    candidates = []
    if isinstance(loaded.get('ros__parameters'), dict):
        candidates.append(loaded['ros__parameters'])
    for section in loaded.values():
        if isinstance(section, dict) and isinstance(section.get('ros__parameters'), dict):
            candidates.append(section['ros__parameters'])

    found: Dict[str, str] = {}
    for params in candidates:
        for key in RESOLVED_KEYS:
            if key in params and params[key] is not None and key not in found:
                found[key] = str(params[key]).strip()
    return found, None, 'ok'


def read_reid_parameters(yaml_path: str) -> Dict[str, str]:
    """Extract the Re-ID keys from a ROS 2 parameter YAML (permissive; never raises).

    Accepts the standard ``<namespace>: {ros__parameters: {...}}`` shape and a flat
    ``ros__parameters`` root. Returns string values for launch ``default_value`` use, and an empty
    dict when the file is missing, malformed, not a mapping, or carries none of the keys.
    """
    found, _error, _status = _read_parameters(yaml_path, strict=False)
    return found


def select_experiment_config(shared_path: str,
                             override: Optional[str] = None) -> Tuple[str, str]:
    """Pick the experiment overlay file, returning ``(path, origin)``.

    ``origin`` is ``'override'`` or ``'none'``. Only an override selects an overlay: launch files
    build their ``DeclareLaunchArgument`` defaults before the CLI is parsed, so a per-run file
    choice cannot be a plain launch argument without giving the two Re-ID channels different
    precedence. The single supported selector is therefore the environment variable, which
    ``robot_bringup.launch.py`` forwards here::

        EDGESYNC_EXPERIMENT_CONFIG=$PKG/share/edgesync_bringup/config/exp7_transreid.yaml \\
            ros2 launch edgesync_bringup robot_bringup.launch.py

    An override that does not exist is not dropped quietly: ``resolve_reid_config`` reads the
    experiment layer strictly and raises.
    """
    candidate = (override if override is not None
                 else os.environ.get(EXPERIMENT_CONFIG_ENV_VAR, ''))
    candidate = str(candidate or '').strip()
    if not candidate:
        return '', 'none'
    if not os.path.isabs(candidate):
        # A relative selection is tried against the shared config's directory first and then
        # against the package directory that contains it, so both 'exp7_transreid.yaml' and
        # 'config/exp7_transreid.yaml' mean the same thing. If neither exists the first form is
        # kept, because resolve_reid_config then reports that exact path as missing.
        shared_dir = os.path.dirname(os.path.abspath(shared_path))
        for base in (shared_dir, os.path.dirname(shared_dir)):
            option = os.path.normpath(os.path.join(base, candidate))
            if os.path.isfile(option):
                return option, 'override'
        return os.path.normpath(os.path.join(shared_dir, candidate)), 'override'
    return candidate, 'override'


def resolve_reid_config(shared_path: str,
                       experiment_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve Re-ID configuration from a shared file plus an optional experiment overlay.

    Returns a provenance record:
        values    – every resolved key as a string, ready for ``default_value``
        origins   – key -> 'shared' | 'experiment' | 'fallback'
        statuses  – file -> loaded / absent / malformed / no_reid_keys
        model_id  – normalized canonical identifier actually selected
        model_source – 'canonical:<layer>' | 'alias:<layer>' | 'fallback'
        model_is_implicit – True when the mock was chosen because nothing configured it
        model_is_valid    – False when the selected id is not a registered backbone
        conflicts – (layer, canonical_value, alias_value) for contradictory explicit input
        errors    – permissive-read messages for files that were not required
    """
    values = dict(HARDCODED_FALLBACKS)
    origins = {key: 'fallback' for key in RESOLVED_KEYS}
    statuses: Dict[str, str] = {}
    conflicts: List[Tuple[str, str, str]] = []
    errors: Dict[str, str] = {}

    layers = [('shared', shared_path, False)]
    if experiment_path and str(experiment_path).strip():
        layers.append(('experiment', str(experiment_path).strip(), True))

    canonical_found: List[Tuple[str, str]] = []
    alias_found: List[Tuple[str, str]] = []

    for layer, path, required in layers:
        found, error, read_status = _read_parameters(path, strict=required)
        if read_status != 'ok':
            statuses[path] = (STATUS_MALFORMED if read_status in ('malformed', 'not_a_mapping')
                              else STATUS_ABSENT)
            if error and not required:
                errors[path] = error
            continue
        statuses[path] = STATUS_LOADED if (set(found) & set(RESOLVED_KEYS)) else STATUS_NO_REID_KEYS
        for key, value in found.items():
            values[key] = value
            origins[key] = layer

        canonical = found.get('reid_model_type')
        alias = found.get(MODEL_ALIAS_KEY)
        if canonical:
            canonical_found.append((layer, canonical))
        if alias:
            alias_found.append((layer, alias))
        # §4.3.4: both names supplied with different values in the same source is a conflict.
        # The documented policy is that the canonical key wins; the conflict is reported, never
        # silently dropped.
        if canonical and alias and normalize_model_id(canonical) != normalize_model_id(alias):
            conflicts.append((layer, canonical, alias))

    if canonical_found:
        layer, selected = canonical_found[-1]
        values['reid_model_type'] = selected
        origins['reid_model_type'] = layer
        model_source = f'canonical:{layer}'
    elif alias_found:
        layer, selected = alias_found[-1]
        values['reid_model_type'] = selected
        origins['reid_model_type'] = layer
        model_source = f'alias:{layer}'
    else:
        values['reid_model_type'] = HARDCODED_FALLBACKS['reid_model_type']
        origins['reid_model_type'] = 'fallback'
        model_source = 'fallback'

    try:
        coerce_camera_id(values['reid_camera_id'])
    except ValueError as exc:
        errors['reid_camera_id'] = str(exc)

    model_id = normalize_model_id(values['reid_model_type'])
    return {
        'values': values,
        'origins': origins,
        'statuses': statuses,
        'model_id': model_id,
        'model_source': model_source,
        'model_is_implicit': model_source == 'fallback',
        'model_is_valid': model_id in known_backbone_ids(),
        'is_mock': model_id in ('osnet_mock',),
        'conflicts': conflicts,
        'errors': errors,
    }


def resolve_reid_launch_defaults(yaml_path: str,
                                 experiment_yaml_path: Optional[str] = None) -> Dict[str, str]:
    """Return launch defaults for every Re-ID key: YAML value where present, else fallback.

    ``reid_model_type`` in the result is the *resolved* selection, i.e. the canonical key when a
    config supplied it and otherwise the guide-name ``reid_model`` alias.
    """
    return resolve_reid_config(yaml_path, experiment_yaml_path)['values']


def describe_reid_sources(yaml_path: str,
                          experiment_yaml_path: Optional[str] = None) -> str:
    """One-line, log-safe summary of where each launch default came from.

    Asset paths configured in the YAML are relative to the installed package share directory;
    identity_node (TR-05) resolves them, so this reports them verbatim rather than claiming a
    CWD-relative path exists.
    """
    record = resolve_reid_config(yaml_path, experiment_yaml_path)
    values, origins = record['values'], record['origins']
    # 'yaml' is the label the TR-01 tests assert on: any value that came from a configuration file
    # rather than HARDCODED_FALLBACKS. Layer names stay distinct so an experiment override is
    # still visible in the log.
    labels = {'shared': 'yaml', 'experiment': 'experiment', 'fallback': 'fallback'}
    parts = []
    for key in RESOLVED_KEYS:
        parts.append(f'{key}={values[key]!r}({labels.get(origins[key], origins[key])})')
    summary = (
        f"Re-ID launch defaults from {os.path.basename(str(yaml_path))}"
        + (f" + {os.path.basename(str(experiment_yaml_path))}"
           if experiment_yaml_path and str(experiment_yaml_path).strip() else '')
        + ": " + ', '.join(parts)
    )
    summary += (
        f" | model_id={record['model_id']!r} source={record['model_source']}"
        f" valid={record['model_is_valid']}"
    )
    if record['is_mock'] and record['model_is_implicit']:
        summary += (
            " | WARNING: no Re-ID model was configured; falling back to the deterministic "
            "'osnet_mock' pseudo-embedding. NOT scientific TransReID output."
        )
    elif record['is_mock']:
        summary += " | NOTE: 'osnet_mock' explicitly selected — pseudo-embeddings only."
    for layer, canonical, alias in record['conflicts']:
        summary += (f" | WARNING: conflicting model ids in layer '{layer}': "
                    f"reid_model_type={canonical!r} vs reid_model={alias!r}; canonical wins.")
    for location, message in record['errors'].items():
        summary += f" | WARNING: {location}: {message}"
    for path, status in record['statuses'].items():
        if status != STATUS_LOADED:
            summary += f" | WARNING: {path} -> {status}"
    return summary
