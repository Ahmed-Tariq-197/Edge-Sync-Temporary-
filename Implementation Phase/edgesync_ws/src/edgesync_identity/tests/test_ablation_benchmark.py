"""Unit tests for the Task 6 benchmark harness and result comparison."""
import json

import numpy as np
import pytest

from edgesync_identity.ablation import benchmark_extractor as bench
from edgesync_identity.ablation import compare_results as cmp
from edgesync_identity.ablation.ablation_config import EXIT_NO_HARDWARE, EXIT_OK, MS_PER_SECOND
from edgesync_identity.ablation.market_dataset import list_records, parse_market_filename
from edgesync_identity.interfaces.base_identity_extractor import BaseIdentityExtractor

STEP_SECONDS = 0.002
WARMUP = 3
ITERATIONS = 20
TOLERANCE = 1e-6


class SteppingClock:
    """Each timed call spans exactly one step; two readings per iteration."""

    def __init__(self, durations):
        self._durations = iter(durations)
        self._now = 0.0
        self._start = True

    def __call__(self):
        if self._start:
            self._start = False
            return self._now
        self._start = True
        self._now += next(self._durations)
        return self._now


def test_measure_calls_discards_warmup_and_reports_percentiles():
    calls = []
    clock = SteppingClock([STEP_SECONDS] * ITERATIONS)
    stats = bench.measure_calls(lambda: calls.append(1), WARMUP, ITERATIONS, clock=clock)
    assert len(calls) == WARMUP + ITERATIONS
    assert stats.samples == ITERATIONS
    assert stats.p50_ms == pytest.approx(STEP_SECONDS * MS_PER_SECOND, abs=TOLERANCE)
    assert stats.p99_ms == pytest.approx(STEP_SECONDS * MS_PER_SECOND, abs=TOLERANCE)
    assert stats.throughput_fps == pytest.approx(1.0 / STEP_SECONDS, abs=1e-3)


def test_summarize_latency_orders_percentiles():
    stats = bench.summarize_latency([0.001 * i for i in range(1, 101)])
    assert stats.min_ms <= stats.p50_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms


def test_parse_market_filename_variants():
    assert parse_market_filename("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market_filename("-1_c3s2_000401_00.jpg") == (-1, 3)
    assert parse_market_filename("0000_c6s3_001001_00.JPG") == (0, 6)
    assert parse_market_filename("Thumbs.db") is None
    assert parse_market_filename("0002_c1s1_000451_03.txt") is None


def test_list_records_filters_and_limits(tmp_path):
    for name in ("0001_c1s1_000001_00.jpg", "0002_c2s1_000002_00.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [r.person_id for r in list_records(str(tmp_path))] == [1, 2]
    assert len(list_records(str(tmp_path), max_images=1)) == 1
    with pytest.raises(FileNotFoundError):
        list_records(str(tmp_path / "absent"))


class IdentityLikeExtractor(BaseIdentityExtractor):
    def extract_embedding(self, image_crop):
        return np.ones(4, dtype=np.float32)


def test_benchmark_latency_without_runner_reports_end_to_end_only():
    result = bench.benchmark_latency(IdentityLikeExtractor(), bench.synthetic_crop(), WARMUP, ITERATIONS)
    assert set(result) == {"end_to_end"}


def test_cli_refuses_mock_embeddings_without_flag(tmp_path):
    output = tmp_path / "out.json"
    code = bench.main(["--backbone", "mobilenetv2", "--mode", "latency", "--output", str(output)])
    assert code == EXIT_NO_HARDWARE
    assert not output.exists()


def test_cli_dry_run_with_allow_mock_writes_schema(tmp_path):
    output = tmp_path / "out.json"
    code = bench.main(["--backbone", "osnet", "--mode", "latency", "--allow-mock", "--warmup", "1",
                       "--iterations", "5", "--output", str(output)])
    assert code == EXIT_OK
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["hardware_backed"] is False
    assert data["latency"]["end_to_end"]["samples"] == 5


def test_mcnemar_exact_known_values():
    assert cmp.mcnemar_exact_p(0, 0) == 1.0
    assert cmp.mcnemar_exact_p(10, 0) == pytest.approx(2 * 0.5 ** 10, abs=TOLERANCE)
    assert cmp.mcnemar_exact_p(5, 5) == 1.0
    assert 0.0 <= cmp.mcnemar_exact_p(3000, 2900) <= 1.0


def test_paired_flags_ignores_unevaluated_queries():
    assert cmp.paired_flags([1, 1, 0, -1], [0, 1, 1, 1]) == (1, 1, 3)
    with pytest.raises(ValueError):
        cmp.paired_flags([1], [1, 0])


def make_result(label, p50, rank1, top1, false_handoff, fh_rate):
    return {
        "label": label, "threshold": 0.7,
        "latency": {"end_to_end": {"p50_ms": p50, "p99_ms": p50 * 1.2, "throughput_fps": 1000.0 / p50}},
        "accuracy": {
            "rank1": rank1, "mAP": rank1 - 0.1, "per_query": {"top1_correct": top1, "false_handoff": false_handoff},
            "cross_robot": {"false_handoff_rate": fh_rate, "false_handoff_rate_ci95": [0.0, 1.0],
                            "pair_false_match_rate": 0.01, "matched_tar_fmr": 0.02,
                            "matched_tar_threshold": 0.6, "equal_error_rate": 0.1},
        },
    }


def test_compare_reports_speedup_drop_and_significant_false_match_increase():
    n = 40
    base = make_result("osnet", 10.0, 0.90, [1] * n, [0] * n, 0.0)
    cand = make_result("mobilenetv2", 4.0, 0.80, [0] * 12 + [1] * (n - 12), [1] * 12 + [0] * (n - 12), 0.3)
    report = cmp.compare(base, cand)
    assert report["speed"]["end_to_end"]["speedup_p50"] == pytest.approx(2.5, abs=TOLERANCE)
    assert report["accuracy"]["rank1"]["drop_pp"] == pytest.approx(10.0, abs=1e-6)
    assert report["accuracy"]["false_handoff"]["discordant"] == [0, 12]
    assert report["verdict"]["inference_faster"] == "supported"
    assert report["verdict"]["false_matches_increase"] == "supported (significant)"
    assert "Verdict" in cmp.to_markdown(report)


def test_compare_marks_not_supported_when_false_matches_do_not_increase():
    n = 10
    base = make_result("osnet", 10.0, 0.90, [1] * n, [1] * n, 0.5)
    cand = make_result("mobilenetv2", 12.0, 0.90, [1] * n, [0] * n, 0.0)
    verdict = cmp.compare(base, cand)["verdict"]
    assert verdict["inference_faster"] == "not supported"
    assert verdict["false_matches_increase"] == "not supported"


def test_markdown_warns_when_runs_are_not_hardware_backed():
    n = 4
    base = make_result("osnet", 10.0, 0.9, [1] * n, [0] * n, 0.0)
    cand = make_result("mobilenetv2", 5.0, 0.9, [1] * n, [0] * n, 0.0)
    assert "WARNING" in cmp.to_markdown(cmp.compare(base, cand))
    base["hardware_backed"] = cand["hardware_backed"] = True
    assert "WARNING" not in cmp.to_markdown(cmp.compare(base, cand))


def test_holm_adjustment_is_monotone_and_order_preserving():
    assert cmp.holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04], abs=TOLERANCE)
    assert cmp.holm_adjust([0.04, 0.01]) == pytest.approx([0.04, 0.02], abs=TOLERANCE)
    assert cmp.holm_adjust([0.4, 0.6]) == pytest.approx([0.8, 0.8], abs=TOLERANCE)
    assert cmp.holm_adjust([0.9]) == [0.9]
