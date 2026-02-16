import numpy as np
import os
from typing import Tuple, Optional, Dict
import logging

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.ecc_utils import ECCWrapper
from src.utils.hash_utils import get_sha256_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuzzyCommitment:
    """
    Fuzzy Commitment Scheme for binding cryptographic keys to biometric embeddings.

    Based on: Juels & Wattenberg (1999)

    Security Properties:
    - Non-invertibility: Helper data reveals nothing about original embedding without key
    - Cryptographic binding: Key cannot be recovered without valid biometric
    - Error tolerance: Reed-Solomon corrects small quantization errors

    Template Format:
    {
        'hash_key': str,           # SHA-256 hash of generated key
        'helper_data': bytes,      # δ = quantized_x ⊕ codeword
        'key_size': int,           # Size of cryptographic key in bytes
    }

    Usage:
        fc = FuzzyCommitment(ecc_capacity=0.2)

        # Enrollment
        hash_key, helper_data = fc.enroll(embedding)

        # Verification
        success, recovered_key = fc.verify(noisy_embedding, hash_key, helper_data)
    """

    def __init__(
        self, key_size: int = 32, ecc_capacity: float = 0.2, embedding_dim: int = 512
    ):
        """
        Initialize Fuzzy Commitment scheme.

        Args:
            key_size: Size of cryptographic key in bytes (default: 32 = 256-bit)
            ecc_capacity: Reed-Solomon error correction capacity (default: 20%)
            embedding_dim: Dimension of biometric embeddings (default: 512)
        """
        self.key_size = key_size
        self.ecc_capacity = ecc_capacity
        self.embedding_dim = embedding_dim

        # Initialize ECC wrapper
        self.ecc = ECCWrapper(
            message_size=key_size, error_capacity_percent=ecc_capacity
        )

        # Calculate padded codeword length (for XOR with embedding)
        # We pad the ECC codeword to match embedding byte length
        self.codeword_length = embedding_dim

        logger.info(
            f"FuzzyCommitment initialized: key_size={key_size}, "
            f"ecc_capacity={ecc_capacity}, embedding_dim={embedding_dim}"
        )

    def _generate_random_key(self) -> bytes:
        """Generate a random cryptographic key."""
        return os.urandom(self.key_size)

    def _pad_codeword(self, codeword: bytes) -> bytes:
        """Pad ECC codeword to embedding dimension for XOR."""
        if len(codeword) >= self.codeword_length:
            return codeword[: self.codeword_length]

        # Pad with zeros
        padded = bytearray(self.codeword_length)
        padded[: len(codeword)] = codeword
        return bytes(padded)

    def _unpad_codeword(self, padded_codeword: bytes) -> bytes:
        """Remove padding from codeword to get original key size."""
        return padded_codeword[: self.key_size]

    def quantize_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Quantize embedding to bytes.

        Args:
            embedding: Biometric embedding (normalized float32 array)

        Returns:
            Quantized bytes (embedding_dim bytes)
        """
        # Ensure embedding is in [-1, 1] range
        embedding = np.clip(embedding, -1.0, 1.0)

        # Convert to uint8 (0-255)
        # -1 → 0, 1 → 255
        quantized = ((embedding + 1.0) * 127.5).astype(np.uint8)

        return bytes(quantized.tobytes())

    def dequantize_embedding(self, quantized: bytes) -> np.ndarray:
        """
        Dequantize bytes back to embedding.

        Args:
            quantized: Quantized bytes

        Returns:
            Dequantized embedding (float32 array)
        """
        arr = np.frombuffer(quantized, dtype=np.uint8)
        # Convert back to [-1, 1]
        embedding = (arr.astype(np.float32) / 127.5) - 1.0
        return embedding[: self.embedding_dim]

    def enroll(self, embedding: np.ndarray) -> Tuple[str, bytes]:
        """
        Enroll a biometric embedding - bind it to a cryptographic key.

        Args:
            embedding: Biometric embedding (512-dim normalized array)

        Returns:
            Tuple of (hash_key, helper_data)
            - hash_key: Hex string of SHA-256(key)
            - helper_data: δ = quantized_x ⊕ codeword
        """
        # Generate random cryptographic key
        key = self._generate_random_key()
        logger.debug(f"Generated random key: {len(key)} bytes")

        # Encode key with Reed-Solomon
        codeword = self.ecc.encode(key)
        logger.debug(f"ECC encoded key: {len(codeword)} bytes")

        # Pad codeword to embedding length
        padded_codeword = self._pad_codeword(codeword)

        # Quantize embedding
        quantized_x = self.quantize_embedding(embedding)

        # Compute helper data: δ = quantized_x ⊕ padded_codeword
        helper_data = bytes(a ^ b for a, b in zip(quantized_x, padded_codeword))

        # Hash the key for verification
        hash_key = get_sha256_hash(key).hex()

        logger.info(f"Enrollment complete: hash_key={hash_key[:16]}...")

        return hash_key, helper_data

    def verify(
        self, embedding: np.ndarray, hash_key: str, helper_data: bytes
    ) -> Tuple[bool, Optional[bytes]]:
        """
        Verify a biometric embedding against stored template.

        Args:
            embedding: Query biometric embedding (512-dim normalized array)
            hash_key: Stored hash of original key (hex string)
            helper_data: Stored helper data δ

        Returns:
            Tuple of (success, recovered_key)
            - success: True if verification successful
            - recovered_key: The recovered cryptographic key (if successful)
        """
        try:
            # Quantize query embedding
            quantized_x_prime = self.quantize_embedding(embedding)

            # Recover candidate codeword: c' = x' ⊕ δ
            candidate_codeword = bytes(
                a ^ b for a, b in zip(quantized_x_prime, helper_data)
            )

            # Extract key bytes from padded codeword
            candidate_key_bytes = self._unpad_codeword(candidate_codeword)

            # Try to decode with Reed-Solomon (error correction)
            recovered_key = self.ecc.decode(candidate_key_bytes)

            # Hash recovered key
            recovered_hash = get_sha256_hash(recovered_key).hex()

            # Verify hash match
            success = recovered_hash == hash_key

            if success:
                logger.info("Verification successful: key recovered")
            else:
                logger.warning("Verification failed: key hash mismatch")

            return success, recovered_key if success else None

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, None

    def get_template_size(self) -> int:
        """Get the size of the protected template in bytes."""
        # hash_key (32 bytes) + helper_data (embedding_dim bytes)
        return 32 + self.embedding_dim

    def __repr__(self) -> str:
        return (
            f"FuzzyCommitment("
            f"key_size={self.key_size}, "
            f"ecc_capacity={self.ecc_capacity}, "
            f"embedding_dim={self.embedding_dim})"
        )


def create_fuzzy_commitment(
    key_size: int = 32, ecc_capacity: float = 0.2, embedding_dim: int = 512
) -> FuzzyCommitment:
    """
    Factory function to create FuzzyCommitment instance.

    Args:
        key_size: Size of cryptographic key in bytes
        ecc_capacity: ECC error correction capacity
        embedding_dim: Dimension of embeddings

    Returns:
        Configured FuzzyCommitment instance
    """
    return FuzzyCommitment(
        key_size=key_size, ecc_capacity=ecc_capacity, embedding_dim=embedding_dim
    )
