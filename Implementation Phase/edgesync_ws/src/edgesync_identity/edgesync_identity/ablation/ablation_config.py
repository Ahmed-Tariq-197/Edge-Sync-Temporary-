"""Named constants and defaults for the Re-ID backbone ablation (no magic numbers)."""
from dataclasses import dataclass
from typing import Tuple

RESULT_SCHEMA_VERSION = 1

# Similarity domain and histogram resolution (bin width = 2 / bins = 0.001).
SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0
HISTOGRAM_BINS = 2000
THRESHOLD_ROUNDING_DIGITS = 6

# Market-1501 conventions.
JUNK_IDENTITY = -1
MARKET_FILENAME_PATTERN = r"^(-?\d+)_c(\d+)s\d+_\d+(?:_\d+)?\.(?:jpg|jpeg|png)$"
CMC_RANKS = (1, 5, 10)

# Latency measurement.
WARMUP_ITERATIONS = 50
MEASURED_ITERATIONS = 1000
LATENCY_PERCENTILES = (50.0, 95.0, 99.0)
MS_PER_SECOND = 1000.0
BYTES_PER_MIB = 1024.0 * 1024.0
SYNTHETIC_CROP_HEIGHT = 288
SYNTHETIC_CROP_WIDTH = 144
SYNTHETIC_CROP_SEED = 0
PIXEL_RANGE_EXCLUSIVE_MAX = 256
PROGRESS_LOG_INTERVAL = 500

# Statistics.
WILSON_Z_95 = 1.96
SIGNIFICANCE_ALPHA = 0.05
PERCENT = 100.0

# Exit codes.
EXIT_OK = 0
EXIT_BAD_ARGUMENTS = 2
EXIT_NO_HARDWARE = 3


@dataclass(frozen=True)
class AblationConfig:
    """Operating point and robot camera split used by the evaluator."""

    similarity_threshold: float = 0.70
    target_true_accept_rate: float = 0.90
    robot_a_cameras: Tuple[int, ...] = (1, 2, 3)
    robot_b_cameras: Tuple[int, ...] = (4, 5, 6)
    histogram_bins: int = HISTOGRAM_BINS
