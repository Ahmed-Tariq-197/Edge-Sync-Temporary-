"""Market-1501 style dataset indexing (file name encodes identity and camera)."""
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .ablation_config import MARKET_FILENAME_PATTERN

_FILENAME_REGEX = re.compile(MARKET_FILENAME_PATTERN, re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    person_id: int
    camera_id: int


def parse_market_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Return (person_id, camera_id) or None when the name does not follow the convention."""
    match = _FILENAME_REGEX.match(filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_records(directory: str, max_images: Optional[int] = None) -> List[ImageRecord]:
    """Index every conforming image in a directory (sorted for reproducibility)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    records: List[ImageRecord] = []
    for name in sorted(os.listdir(directory)):
        parsed = parse_market_filename(name)
        if parsed is None:
            continue
        records.append(ImageRecord(os.path.join(directory, name), parsed[0], parsed[1]))
        if max_images is not None and len(records) >= max_images:
            break
    return records


def load_rgb(path: str) -> np.ndarray:
    """Read an image as an HxWx3 uint8 RGB array."""
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
