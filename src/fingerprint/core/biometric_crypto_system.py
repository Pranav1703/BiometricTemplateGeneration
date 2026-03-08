import numpy as np
import os
from typing import Tuple, Optional, Dict
import logging

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.fingerprint.core.cancelable_transform import CancelableTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricCryptoSystem:
    """
    Complete biometric cryptosystem with cancelable transform + PBKDF2 key derivation.

    Architecture:
    Raw Embedding → Cancelable Transform → PBKDF2 → Protected Template

    Note: Uses similarity-based verification (cosine similarity) which is robust
    to biometric variation, unlike fuzzy commitment which fails with high-dim embeddings.

    Security Properties:
    1. Cancelable: Revocable templates, unlinkable across applications
    2. PBKDF2: Cryptographic key derivation from transformed embedding
    3. Non-invertibility: Without user_key, original cannot be recovered

    Template Format:
    {
        'transformed': np.ndarray,    # Transformed embedding
        'salt': bytes,               # Random salt for PBKDF2
        'key_hash': bytes,           # SHA-256 hash of derived key
        'cancelable_params': dict    # For verification
    }

    Usage:
        >>> system = BiometricCryptoSystem(embedding_dim=512, cancelable_alpha=0.6)
        >>> template = system.enroll(embedding, "user_001")
        >>> success, key = system.verify(query_embedding, template, "user_001")
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        key_size: int = 32,
        cancelable_alpha: float = 0.6,
    ):
        """
        Initialize the biometric cryptosystem.

        Args:
            embedding_dim: Dimension of biometric embeddings (default: 512)
            key_size: Size of cryptographic key in bytes (default: 32 = 256-bit)
            cancelable_alpha: Cancelable transform blending parameter (default: 0.6)
        """
        self.embedding_dim = embedding_dim
        self.key_size = key_size
        self.cancelable_alpha = cancelable_alpha

        self.cancelable = CancelableTransform(embedding_dim, alpha=cancelable_alpha)

        logger.info("=" * 60)
        logger.info("Biometric CryptoSystem initialized")
        logger.info("=" * 60)
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Key size: {key_size * 8} bits")
        logger.info(f"  Cancelable alpha: {cancelable_alpha}")
        logger.info("  Method: Cancelable + PBKDF2")
        logger.info("=" * 60)

    def enroll(self, raw_embedding: np.ndarray, user_key: str) -> Dict:
        """
        Enroll biometric template with full crypto system.

        Args:
            raw_embedding: Raw embedding from deep learning model (512-dim)
            user_key: User/application-specific key for cancelable transform

        Returns:
            Protected template dictionary

        Example:
            >>> system = BiometricCryptoSystem()
            >>> embedding = backbone(img)
            >>> template = system.enroll(embedding, "user_001")
            >>> print(template.keys())
        """
        logger.info(f"Enrolling template for user_key: {user_key}")

        if TORCH_AVAILABLE and hasattr(raw_embedding, "detach"):
            raw_embedding = raw_embedding.detach().cpu().numpy()
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)

        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding = raw_embedding / norm

        template, cancelable_params = self.cancelable.enroll_with_key(
            raw_embedding, user_key
        )

        template["cancelable_params"] = cancelable_params

        logger.info("Enrollment completed successfully")

        return template

    def verify(
        self, raw_embedding: np.ndarray, template: Dict, user_key: str
    ) -> Tuple[bool, Optional[bytes]]:
        """
        Verify biometric template with full crypto system.

        Args:
            raw_embedding: Query embedding from deep learning model (512-dim)
            template: Protected template from enrollment
            user_key: User/application-specific key (same as enrollment)

        Returns:
            (success, recovered_key)

            - success: True if authentication succeeded
            - recovered_key: 256-bit cryptographic key if successful
        """
        logger.info(f"Verifying template for user_key: {user_key}")

        if TORCH_AVAILABLE and hasattr(raw_embedding, "detach"):
            raw_embedding = raw_embedding.detach().cpu().numpy()
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)

        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding = raw_embedding / norm

        success, recovered_key = self.cancelable.verify_with_key(
            raw_embedding, template, template["cancelable_params"]
        )

        if success:
            logger.info("Verification SUCCESSFUL")
        else:
            logger.info("Verification FAILED")

        return success, recovered_key

    def cancel_and_reissue(
        self, raw_embedding: np.ndarray, old_user_key: str, new_user_key: str
    ) -> Dict:
        """
        Cancel old template and issue new one with different user key.

        Args:
            raw_embedding: Biometric embedding (can use enrollment sample)
            old_user_key: Old user key to cancel
            new_user_key: New user key to issue

        Returns:
            New protected template dictionary
        """
        logger.info(
            f"Cancelling old template ({old_user_key}) and issuing new ({new_user_key})"
        )

        new_template, new_cancelable_params = self.cancelable.enroll_with_key(
            raw_embedding, new_user_key
        )
        new_template["cancelable_params"] = new_cancelable_params

        logger.info("New template issued successfully")

        return new_template

    def get_system_info(self) -> Dict:
        """Get system configuration and parameters."""
        return {
            "embedding_dim": self.embedding_dim,
            "key_size": self.key_size,
            "cancelable_alpha": self.cancelable_alpha,
            "method": "Cancelable + PBKDF2",
        }
