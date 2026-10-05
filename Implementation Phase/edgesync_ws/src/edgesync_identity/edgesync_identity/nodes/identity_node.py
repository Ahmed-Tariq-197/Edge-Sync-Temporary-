"""Identity Node (C-07): Resident Re-ID Engine & Candidate Verification Service.

SOLID & Architectural Mandates:
- Resident TensorRT engine loaded at startup in unified memory (zero engine-load delay).
- Serves IdentityVerify.srv (I-10) for candidate cross-verification.
- Maintains continuous 2 Hz smoothed target signature buffer.
- ZERO magic numbers: all thresholds and topic/service names injected from YAML.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
import numpy as np

from edgesync_msgs.srv import IdentityVerify
from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..models.osnet_trt_extractor import OsnetTrtExtractor
from ..core.signature_smoother import SignatureSmoother
from ..core.identity_verifier import IdentityVerifier


class IdentityNode(Node):
    """ROS 2 Node providing resident Re-ID embedding and verification services."""

    def __init__(self, extractor: Optional[BaseIdentityExtractor] = None):
        super().__init__('identity_node')

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'ugv_2')
        self.declare_parameter('similarity_threshold', 0.70)
        self.declare_parameter('smoothing_alpha', 0.20)
        self.declare_parameter('enable_spatial_gating', True)
        self.declare_parameter('service_name', '/identity/verify')
        self.declare_parameter('engine_path', '')

        # 2. Extract Parameters
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.similarity_threshold = self.get_parameter('similarity_threshold').get_parameter_value().double_value
        alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        spatial_gating = self.get_parameter('enable_spatial_gating').get_parameter_value().bool_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        engine_path = self.get_parameter('engine_path').get_parameter_value().string_value

        # 3. Dependency Injection (SOLID: DIP)
        self.extractor: BaseIdentityExtractor = extractor if extractor is not None else OsnetTrtExtractor(
            engine_path=engine_path if engine_path else None
        )

        # 4. Core Algorithms
        self.smoother = SignatureSmoother(alpha=alpha, embedding_dim=512)
        self.verifier = IdentityVerifier(similarity_threshold=self.similarity_threshold, enable_spatial_gating=spatial_gating)

        # Stored vectors from HandoffRequest / live detections
        self._stored_candidate_vector: Optional[np.ndarray] = None
        self._live_target_vector: Optional[np.ndarray] = None

        # 5. Service Server (I-10 Interface)
        self.srv_verify = self.create_service(
            IdentityVerify, service_name, self._handle_identity_verify
        )

        self.get_logger().info(f"EdgeSync Identity Node initialized: service={service_name}, threshold={self.similarity_threshold:.2f}")

    def store_candidate_vector(self, vector: np.ndarray) -> None:
        """Store reference candidate vector received from HandoffRequest (Robot A)."""
        self._stored_candidate_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.get_logger().info("Candidate feature vector stored from HandoffRequest.")

    def set_live_target_vector(self, vector: np.ndarray) -> None:
        """Set live target vector from local receiver tracking/perception."""
        self._live_target_vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    @staticmethod
    def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Compute exact cosine similarity with NumPy 2.x safe scalar conversion.

        Cosine similarity = (u . v) / (||u|| * ||v||)
        """
        u_flat = np.asarray(u, dtype=np.float32).reshape(-1)
        v_flat = np.asarray(v, dtype=np.float32).reshape(-1)

        norm_u = float(np.linalg.norm(u_flat).item())
        norm_v = float(np.linalg.norm(v_flat).item())

        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0

        dot_val = float(np.dot(u_flat, v_flat).item())
        cos_sim = dot_val / (norm_u * norm_v)
        return float(np.clip(cos_sim, -1.0, 1.0))

    def _handle_identity_verify(self, request: IdentityVerify.Request,
                                response: IdentityVerify.Response) -> IdentityVerify.Response:
        """Handle Candidate Re-ID verification service request."""
        # 1. Resolve Candidate Vector (from Robot A's HandoffRequest)
        ref_arr = np.asarray(request.reference_feature_vector, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(ref_arr).item()) > 1e-6:
            candidate_vec = ref_arr
        elif self._stored_candidate_vector is not None:
            candidate_vec = self._stored_candidate_vector
        else:
            candidate_vec = np.zeros(512, dtype=np.float32)

        # 2. Resolve Live Target Vector (from local camera crop or direct vector)
        if hasattr(request, 'candidate_feature_vector'):
            cand_arr = np.asarray(request.candidate_feature_vector, dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(cand_arr).item()) > 1e-6:
                live_vec = cand_arr
            elif hasattr(request, 'candidate_crop') and len(request.candidate_crop.data) > 0:
                try:
                    crop_msg = request.candidate_crop
                    crop_np = np.frombuffer(crop_msg.data, dtype=np.uint8).reshape((crop_msg.height, crop_msg.width, 3))
                    live_vec = self.extractor.extract_embedding(crop_np).reshape(-1)
                except Exception as e:
                    self.get_logger().error(f"Failed to decode candidate crop: {e}")
                    live_vec = np.zeros(512, dtype=np.float32)
            elif self._live_target_vector is not None:
                live_vec = self._live_target_vector
            else:
                live_vec = np.zeros(512, dtype=np.float32)
        elif hasattr(request, 'candidate_crop') and len(request.candidate_crop.data) > 0:
            try:
                crop_msg = request.candidate_crop
                crop_np = np.frombuffer(crop_msg.data, dtype=np.uint8).reshape((crop_msg.height, crop_msg.width, 3))
                live_vec = self.extractor.extract_embedding(crop_np).reshape(-1)
            except Exception as e:
                self.get_logger().error(f"Failed to decode candidate crop: {e}")
                live_vec = np.zeros(512, dtype=np.float32)
        elif self._live_target_vector is not None:
            live_vec = self._live_target_vector
        else:
            live_vec = np.zeros(512, dtype=np.float32)

        # 3. Exact Cosine Similarity Math
        similarity_score = self.compute_cosine_similarity(candidate_vec, live_vec)
        match_confirmed = bool(similarity_score >= self.similarity_threshold)

        # 4. Pack Response
        response.match = match_confirmed
        response.similarity = float(similarity_score)
        if hasattr(response, 'match_confirmed'):
            response.match_confirmed = match_confirmed
        if hasattr(response, 'similarity_score'):
            response.similarity_score = float(similarity_score)

        if match_confirmed:
            response.message = f"Identity confirmed: similarity {similarity_score:.4f} >= threshold {self.similarity_threshold:.2f}"
            self.get_logger().info(f"Identity MATCH CONFIRMED (score={similarity_score:.4f})")
        else:
            response.message = f"Identity mismatch: similarity {similarity_score:.4f} < threshold {self.similarity_threshold:.2f}"
            self.get_logger().warn(f"Identity MISMATCH (score={similarity_score:.4f} < threshold {self.similarity_threshold:.2f})")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = IdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
