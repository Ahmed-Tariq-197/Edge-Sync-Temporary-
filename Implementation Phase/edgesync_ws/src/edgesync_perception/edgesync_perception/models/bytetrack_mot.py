"""ByteTrack Multi-Object Tracker implementation.

Implements BaseTracker (SOLID: DIP & OCP) using two-stage IoU matching.
"""
from typing import List, Dict
import numpy as np
from ..interfaces.base_tracker import BaseTracker, Track
from ..interfaces.base_detector import Detection


def compute_iou(box1: tuple[float, float, float, float],
                box2: tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class ByteTrackMOT(BaseTracker):
    """Multi-object tracker associating high and low confidence detections."""

    def __init__(self, track_high_thresh: float = 0.50,
                 track_low_thresh: float = 0.20,
                 match_iou_thresh: float = 0.40,
                 max_time_lost: int = 30):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_time_lost = max_time_lost
        self._next_id = 1
        self._active_tracks: Dict[int, Track] = {}
        self._lost_frames: Dict[int, int] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks using detections."""
        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh]

        updated_tracks: List[Track] = []
        matched_det_indices = set()
        matched_track_ids = set()

        # Step 1: Match high confidence detections with active tracks
        for track_id, track in list(self._active_tracks.items()):
            best_iou = self.match_iou_thresh
            best_idx = -1
            for idx, det in enumerate(high_dets):
                if idx in matched_det_indices:
                    continue
                iou = compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1:
                det = high_dets[best_idx]
                matched_det_indices.add(best_idx)
                matched_track_ids.add(track_id)
                self._active_tracks[track_id] = Track(
                    track_id=track_id,
                    xmin=det.xmin, ymin=det.ymin,
                    xmax=det.xmax, ymax=det.ymax,
                    confidence=det.confidence,
                    is_activated=True
                )
                self._lost_frames[track_id] = 0
                updated_tracks.append(self._active_tracks[track_id])

        # Step 2: Match remaining tracks with low confidence detections
        unmatched_track_ids = set(self._active_tracks.keys()) - matched_track_ids
        for track_id in unmatched_track_ids:
            track = self._active_tracks[track_id]
            best_iou = self.match_iou_thresh
            best_low_idx = -1
            for idx, det in enumerate(low_dets):
                iou = compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_low_idx = idx

            if best_low_idx != -1:
                det = low_dets[best_low_idx]
                self._active_tracks[track_id] = Track(
                    track_id=track_id,
                    xmin=det.xmin, ymin=det.ymin,
                    xmax=det.xmax, ymax=det.ymax,
                    confidence=det.confidence,
                    is_activated=True
                )
                self._lost_frames[track_id] = 0
                updated_tracks.append(self._active_tracks[track_id])
            else:
                self._lost_frames[track_id] = self._lost_frames.get(track_id, 0) + 1
                if self._lost_frames[track_id] > self.max_time_lost:
                    del self._active_tracks[track_id]
                    del self._lost_frames[track_id]

        # Step 3: Initialize new tracks for unmatched high confidence detections
        for idx, det in enumerate(high_dets):
            if idx not in matched_det_indices:
                new_id = self._next_id
                self._next_id += 1
                new_track = Track(
                    track_id=new_id,
                    xmin=det.xmin, ymin=det.ymin,
                    xmax=det.xmax, ymax=det.ymax,
                    confidence=det.confidence,
                    is_activated=True
                )
                self._active_tracks[new_id] = new_track
                self._lost_frames[new_id] = 0
                updated_tracks.append(new_track)

        return updated_tracks
