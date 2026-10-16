"""Compare a baseline (OSNet) and a candidate (MobileNetV2) benchmark result.

Reports Rank-1 drop vs. inference speed gain and tests the Task 6 hypothesis:
inference becomes faster, but false cross-robot matches may increase.
"""
import argparse
import json
import math
import sys
from typing import List, Optional, Tuple

import numpy as np

from .ablation_config import EXIT_OK, PERCENT, SIGNIFICANCE_ALPHA
from .reid_metrics import NOT_EVALUATED

LATENCY_STAGES = ("end_to_end", "engine_only")


def mcnemar_exact_p(base_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    n = base_only + candidate_only
    if n == 0:
        return 1.0
    k = min(base_only, candidate_only)
    log_half_n = n * math.log(0.5)
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
               for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for position, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - position) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def paired_flags(base: List[int], candidate: List[int]) -> Tuple[int, int, int]:
    """Return (base_only, candidate_only, paired_total) over queries evaluated in both runs."""
    a, b = np.asarray(base), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Per-query arrays differ in length: {a.shape} vs {b.shape}")
    both = (a != NOT_EVALUATED) & (b != NOT_EVALUATED)
    return int(((a == 1) & (b == 0) & both).sum()), int(((a == 0) & (b == 1) & both).sum()), int(both.sum())


def compare(base: dict, cand: dict) -> dict:
    out: dict = {"baseline": base["label"], "candidate": cand["label"], "threshold": base["threshold"],
                 "hardware_backed": bool(base.get("hardware_backed") and cand.get("hardware_backed"))}
    if "latency" in base and "latency" in cand:
        speed = {}
        for stage in LATENCY_STAGES:
            if stage in base["latency"] and stage in cand["latency"]:
                b, c = base["latency"][stage], cand["latency"][stage]
                speed[stage] = {
                    "baseline_p50_ms": b["p50_ms"], "candidate_p50_ms": c["p50_ms"],
                    "baseline_p99_ms": b["p99_ms"], "candidate_p99_ms": c["p99_ms"],
                    "speedup_p50": b["p50_ms"] / c["p50_ms"] if c["p50_ms"] > 0 else float("inf"),
                    "baseline_fps": b["throughput_fps"], "candidate_fps": c["throughput_fps"],
                }
        out["speed"] = speed
        out["engine_size_mib"] = {"baseline": base.get("engine_size_mib"),
                                  "candidate": cand.get("engine_size_mib")}
    if "accuracy" in base and "accuracy" in cand:
        ba, ca = base["accuracy"], cand["accuracy"]
        b_only, c_only, paired = paired_flags(ba["per_query"]["top1_correct"], ca["per_query"]["top1_correct"])
        rank1 = {"baseline": ba["rank1"], "candidate": ca["rank1"],
                 "drop_pp": (ba["rank1"] - ca["rank1"]) * PERCENT,
                 "mcnemar_p": mcnemar_exact_p(b_only, c_only), "discordant": [b_only, c_only]}
        fb_only, fc_only, _ = paired_flags(ba["per_query"]["false_handoff"], ca["per_query"]["false_handoff"])
        bx, cx = ba["cross_robot"], ca["cross_robot"]
        fh_p = mcnemar_exact_p(fb_only, fc_only)
        rank1["mcnemar_p_holm"], fh_p_holm = holm_adjust([rank1["mcnemar_p"], fh_p])
        out["accuracy"] = {
            "rank1": rank1,
            "mAP": {"baseline": ba["mAP"], "candidate": ca["mAP"],
                    "drop_pp": (ba["mAP"] - ca["mAP"]) * PERCENT},
            "false_handoff": {
                "baseline": bx["false_handoff_rate"], "candidate": cx["false_handoff_rate"],
                "baseline_ci95": bx["false_handoff_rate_ci95"], "candidate_ci95": cx["false_handoff_rate_ci95"],
                "delta_pp": (cx["false_handoff_rate"] - bx["false_handoff_rate"]) * PERCENT,
                "mcnemar_p": fh_p, "mcnemar_p_holm": fh_p_holm, "discordant": [fb_only, fc_only]},
            "pair_false_match_rate": {"baseline": bx["pair_false_match_rate"],
                                      "candidate": cx["pair_false_match_rate"]},
            "matched_tar_fmr": {"baseline": bx["matched_tar_fmr"], "candidate": cx["matched_tar_fmr"],
                                "baseline_threshold": bx["matched_tar_threshold"],
                                "candidate_threshold": cx["matched_tar_threshold"]},
            "equal_error_rate": {"baseline": bx["equal_error_rate"], "candidate": cx["equal_error_rate"]},
        }
    out["verdict"] = verdict(out)
    return out


def verdict(report: dict) -> dict:
    result = {}
    speed = report.get("speed", {}).get("end_to_end")
    if speed:
        faster = speed["candidate_p50_ms"] < speed["baseline_p50_ms"]
        result["inference_faster"] = "supported" if faster else "not supported"
    fh = report.get("accuracy", {}).get("false_handoff")
    if fh:
        higher = fh["candidate"] > fh["baseline"]
        significant = fh["mcnemar_p_holm"] < SIGNIFICANCE_ALPHA
        if higher and significant:
            result["false_matches_increase"] = "supported (significant)"
        elif higher:
            result["false_matches_increase"] = "directional only (not significant)"
        else:
            result["false_matches_increase"] = "not supported"
    return result


def to_markdown(report: dict) -> str:
    lines = [f"## Task 6 comparison: {report['baseline']} vs {report['candidate']}",
             f"Operating threshold: {report['threshold']:.2f}", ""]
    if not report["hardware_backed"]:
        lines += ["WARNING: at least one run used mock embeddings; these numbers are NOT valid results.", ""]
    if "speed" in report:
        lines += ["| Stage | Base p50 (ms) | Cand p50 (ms) | Speedup | Base p99 (ms) | Cand p99 (ms) | Cand FPS |",
                  "|---|---|---|---|---|---|---|"]
        for stage, s in report["speed"].items():
            lines.append(f"| {stage} | {s['baseline_p50_ms']:.2f} | {s['candidate_p50_ms']:.2f} | "
                         f"{s['speedup_p50']:.2f}x | {s['baseline_p99_ms']:.2f} | "
                         f"{s['candidate_p99_ms']:.2f} | {s['candidate_fps']:.1f} |")
        lines.append("")
    if "accuracy" in report:
        a = report["accuracy"]
        fh, r1 = a["false_handoff"], a["rank1"]
        lines += ["| Metric | Baseline | Candidate | Delta | McNemar p (Holm) |", "|---|---|---|---|---|",
                  f"| Rank-1 | {r1['baseline']:.4f} | {r1['candidate']:.4f} | {-r1['drop_pp']:+.2f} pp | {r1['mcnemar_p_holm']:.4g} |",
                  f"| mAP | {a['mAP']['baseline']:.4f} | {a['mAP']['candidate']:.4f} | {-a['mAP']['drop_pp']:+.2f} pp | n/a |",
                  f"| False-handoff rate | {fh['baseline']:.4f} | {fh['candidate']:.4f} | {fh['delta_pp']:+.2f} pp | {fh['mcnemar_p_holm']:.4g} |",
                  f"| Pair FMR at threshold | {a['pair_false_match_rate']['baseline']:.5f} | {a['pair_false_match_rate']['candidate']:.5f} | n/a | n/a |",
                  f"| FMR at matched TAR | {a['matched_tar_fmr']['baseline']:.5f} | {a['matched_tar_fmr']['candidate']:.5f} | n/a | n/a |",
                  f"| EER | {a['equal_error_rate']['baseline']:.4f} | {a['equal_error_rate']['candidate']:.4f} | n/a | n/a |",
                  ""]
    lines.append("Verdict: " + "; ".join(f"{k} = {v}" for k, v in report["verdict"].items()))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two Re-ID benchmark results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args(argv)
    with open(args.baseline, encoding="utf-8") as h1, open(args.candidate, encoding="utf-8") as h2:
        report = compare(json.load(h1), json.load(h2))
    markdown = to_markdown(report)
    print(markdown)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
