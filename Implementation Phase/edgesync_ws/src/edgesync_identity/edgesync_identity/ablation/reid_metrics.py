"""Re-ID evaluation: CMC/Rank-k, mAP and cross-robot false-match metrics (pure NumPy).

Protocol (Market-1501 style):
- Junk gallery items (identity -1) are ignored.
- Gallery items with the query's identity AND camera are ignored for standard CMC/mAP.
- AP is the mean of precision values at each true match rank (non-interpolated).

Cross-robot protocol: cameras are split into two robot groups. Each query is compared
only against gallery items seen by the OTHER robot group, which mirrors the handoff.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .ablation_config import (
    CMC_RANKS, JUNK_IDENTITY, SIMILARITY_MAX, SIMILARITY_MIN,
    THRESHOLD_ROUNDING_DIGITS, WILSON_Z_95, AblationConfig,
)

NOT_EVALUATED = -1


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return 0.0, 0.0
    p_hat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p_hat + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / trials + z2 / (4.0 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class SimilarityHistogram:
    """Accumulates genuine and impostor similarity histograms for ROC-style queries."""

    def __init__(self, bins: int):
        self.bins = bins
        self.edges = np.linspace(SIMILARITY_MIN, SIMILARITY_MAX, bins + 1)
        self.width = (SIMILARITY_MAX - SIMILARITY_MIN) / bins
        self.genuine = np.zeros(bins, dtype=np.int64)
        self.impostor = np.zeros(bins, dtype=np.int64)

    def add(self, genuine_sims: np.ndarray, impostor_sims: np.ndarray) -> None:
        if genuine_sims.size:
            clipped = np.clip(genuine_sims, SIMILARITY_MIN, SIMILARITY_MAX)
            self.genuine += np.histogram(clipped, bins=self.edges)[0]
        if impostor_sims.size:
            clipped = np.clip(impostor_sims, SIMILARITY_MIN, SIMILARITY_MAX)
            self.impostor += np.histogram(clipped, bins=self.edges)[0]

    def _bin_index(self, threshold: float) -> int:
        raw = round((threshold - SIMILARITY_MIN) / self.width, THRESHOLD_ROUNDING_DIGITS)
        return int(min(max(math.ceil(raw), 0), self.bins))

    @staticmethod
    def _rate(counts: np.ndarray, index: int) -> float:
        total = int(counts.sum())
        return float(counts[index:].sum()) / total if total else 0.0

    def true_accept_rate(self, threshold: float) -> float:
        return self._rate(self.genuine, self._bin_index(threshold))

    def false_match_rate(self, threshold: float) -> float:
        return self._rate(self.impostor, self._bin_index(threshold))

    def _rates_per_bin(self) -> Tuple[np.ndarray, np.ndarray]:
        genuine_total = max(int(self.genuine.sum()), 1)
        impostor_total = max(int(self.impostor.sum()), 1)
        tar = np.cumsum(self.genuine[::-1])[::-1] / genuine_total
        fmr = np.cumsum(self.impostor[::-1])[::-1] / impostor_total
        return tar, fmr

    def threshold_at_true_accept_rate(self, target: float) -> Tuple[float, float]:
        """Highest threshold whose TAR >= target; returns (threshold, FMR at that threshold)."""
        tar, fmr = self._rates_per_bin()
        eligible = np.flatnonzero(tar >= target)
        index = int(eligible[-1]) if eligible.size else 0
        return float(self.edges[index]), float(fmr[index])

    def equal_error_rate(self) -> Tuple[float, float]:
        """Return (EER, threshold) where false-match and false-reject rates cross."""
        tar, fmr = self._rates_per_bin()
        frr = 1.0 - tar
        index = int(np.argmin(np.abs(fmr - frr)))
        return float((fmr[index] + frr[index]) / 2.0), float(self.edges[index])


@dataclass
class ReidEvaluation:
    """Aggregate and per-query results of one backbone evaluation."""

    num_queries: int = 0
    cmc: Dict[int, float] = field(default_factory=dict)
    mean_ap: float = 0.0
    cross_robot_queries: int = 0
    cross_robot_fmr: float = 0.0
    cross_robot_tar: float = 0.0
    false_handoff_count: int = 0
    correct_handoff_count: int = 0
    equal_error_rate: float = 0.0
    equal_error_threshold: float = 0.0
    matched_tar_threshold: float = 0.0
    matched_tar_fmr: float = 0.0
    top1_correct: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    false_handoff: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    @property
    def false_handoff_rate(self) -> float:
        return self.false_handoff_count / self.cross_robot_queries if self.cross_robot_queries else 0.0

    @property
    def correct_handoff_rate(self) -> float:
        return self.correct_handoff_count / self.cross_robot_queries if self.cross_robot_queries else 0.0

    def to_dict(self) -> dict:
        low, high = wilson_interval(self.false_handoff_count, self.cross_robot_queries)
        return {
            "num_queries": self.num_queries,
            "rank1": self.cmc.get(1, 0.0),
            "rank5": self.cmc.get(5, 0.0),
            "rank10": self.cmc.get(10, 0.0),
            "mAP": self.mean_ap,
            "cross_robot": {
                "queries": self.cross_robot_queries,
                "pair_false_match_rate": self.cross_robot_fmr,
                "pair_true_accept_rate": self.cross_robot_tar,
                "false_handoff_count": self.false_handoff_count,
                "false_handoff_rate": self.false_handoff_rate,
                "false_handoff_rate_ci95": [low, high],
                "correct_handoff_count": self.correct_handoff_count,
                "correct_handoff_rate": self.correct_handoff_rate,
                "equal_error_rate": self.equal_error_rate,
                "equal_error_threshold": self.equal_error_threshold,
                "matched_tar_threshold": self.matched_tar_threshold,
                "matched_tar_fmr": self.matched_tar_fmr,
            },
            "per_query": {
                "top1_correct": self.top1_correct.astype(int).tolist(),
                "false_handoff": self.false_handoff.astype(int).tolist(),
            },
        }


class ReidEvaluator:
    """Computes standard and cross-robot Re-ID metrics from L2-normalised embeddings."""

    def __init__(self, config: AblationConfig = AblationConfig()):
        self.config = config

    def evaluate(self, query_emb: np.ndarray, query_pid: np.ndarray, query_cam: np.ndarray,
                 gallery_emb: np.ndarray, gallery_pid: np.ndarray,
                 gallery_cam: np.ndarray) -> ReidEvaluation:
        cfg = self.config
        query_emb = np.asarray(query_emb, dtype=np.float32)
        gallery_emb = np.asarray(gallery_emb, dtype=np.float32)
        num_queries = query_emb.shape[0]
        not_junk = gallery_pid != JUNK_IDENTITY
        in_a = np.isin(gallery_cam, cfg.robot_a_cameras)
        in_b = np.isin(gallery_cam, cfg.robot_b_cameras)

        histogram = SimilarityHistogram(cfg.histogram_bins)
        top1 = np.full(num_queries, NOT_EVALUATED, dtype=np.int8)
        false_handoff = np.full(num_queries, NOT_EVALUATED, dtype=np.int8)
        cmc_hits = {rank: 0 for rank in CMC_RANKS}
        ap_values: List[float] = []
        valid_queries = 0
        correct_handoffs = 0

        for qi in range(num_queries):
            qpid, qcam = int(query_pid[qi]), int(query_cam[qi])
            sims = gallery_emb @ query_emb[qi]

            valid = not_junk & ~((gallery_pid == qpid) & (gallery_cam == qcam))
            indices = np.flatnonzero(valid)
            matches = gallery_pid[indices] == qpid
            if matches.any():
                order = np.argsort(-sims[indices], kind="stable")
                hits = matches[order]
                ranks = np.flatnonzero(hits) + 1
                valid_queries += 1
                top1[qi] = int(hits[0])
                for rank in CMC_RANKS:
                    cmc_hits[rank] += int(ranks[0] <= rank)
                ap_values.append(float(np.mean(np.arange(1, ranks.size + 1) / ranks)))

            if qcam in cfg.robot_a_cameras:
                other = in_b
            elif qcam in cfg.robot_b_cameras:
                other = in_a
            else:
                continue
            cross = np.flatnonzero(not_junk & other)
            if cross.size == 0:
                continue
            cross_sims = sims[cross]
            is_genuine = gallery_pid[cross] == qpid
            histogram.add(cross_sims[is_genuine], cross_sims[~is_genuine])
            best = int(np.argmax(cross_sims))
            accepted = cross_sims[best] >= cfg.similarity_threshold
            wrong = not is_genuine[best]
            false_handoff[qi] = int(accepted and wrong)
            correct_handoffs += int(accepted and not wrong)

        evaluated = false_handoff != NOT_EVALUATED
        eer, eer_threshold = histogram.equal_error_rate()
        matched_threshold, matched_fmr = histogram.threshold_at_true_accept_rate(
            cfg.target_true_accept_rate)
        denom = max(valid_queries, 1)
        return ReidEvaluation(
            num_queries=valid_queries,
            cmc={rank: cmc_hits[rank] / denom for rank in CMC_RANKS},
            mean_ap=float(np.mean(ap_values)) if ap_values else 0.0,
            cross_robot_queries=int(evaluated.sum()),
            cross_robot_fmr=histogram.false_match_rate(cfg.similarity_threshold),
            cross_robot_tar=histogram.true_accept_rate(cfg.similarity_threshold),
            false_handoff_count=int((false_handoff == 1).sum()),
            correct_handoff_count=correct_handoffs,
            equal_error_rate=eer,
            equal_error_threshold=eer_threshold,
            matched_tar_threshold=matched_threshold,
            matched_tar_fmr=matched_fmr,
            top1_correct=top1,
            false_handoff=false_handoff,
        )
