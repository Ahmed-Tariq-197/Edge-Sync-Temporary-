"""YOLOv8 Object Detector implementation using Ultralytics format (SOLID: DIP & OCP).

Follows BaseDetector contract and allows transparent swapping with TRT or Mock detectors.
"""
from typing import List, Optional, Any
import numpy as np
from ..interfaces.base_detector import BaseDetector, Detection


class YoloV8Detector(BaseDetector):
    """YOLOv8 object detector using Ultralytics inference format."""

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.40,
                 target_class_id: int = 0,
                 model_instance: Optional[Any] = None):
        """Initialize YOLOv8 detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt) or export.
            confidence_threshold: Minimum confidence score for detection.
            target_class_id: Target class index (0 for 'person' in COCO).
            model_instance: Optional pre-instantiated YOLO model (for testing/mocking).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_class_id = target_class_id
        self._model = model_instance
        self._model_load_attempted = model_instance is not None

        if self._model is None:
            self._init_model()

    def _init_model(self):
        """Lazy load YOLO model if ultralytics is available."""
        self._model_load_attempted = True
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        except Exception:
            # Model file missing or ultralytics not installed (e.g. host test environment)
            self._model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Execute 2D object detection on input RGB frame.

        Args:
            image: HxWx3 uint8 numpy array in RGB format.

        Returns:
            List of Detection objects filtered by class and confidence.
        """
        if self._model is None and not self._model_load_attempted:
            self._init_model()

        if self._model is None:
            return []

        try:
            # Inference call following Ultralytics YOLO format
            results = self._model(
                image,
                conf=self.confidence_threshold,
                classes=[self.target_class_id],
                verbose=False
            )
            detections: List[Detection] = []
            if not results:
                return detections

            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                return detections

            for box in res.boxes:
                # Handle both PyTorch tensors and numpy array representations
                if hasattr(box.xyxy, 'cpu'):
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy().item())
                    cls_id = int(box.cls[0].cpu().numpy().item())
                else:
                    coords = np.asarray(box.xyxy[0]).tolist()
                    conf = float(np.asarray(box.conf[0]).item())
                    cls_id = int(np.asarray(box.cls[0]).item())

                detections.append(Detection(
                    xmin=float(coords[0]),
                    ymin=float(coords[1]),
                    xmax=float(coords[2]),
                    ymax=float(coords[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name="person"
                ))

            return detections
        except Exception:
            return []
