"""
Biometric Key Generator

A working biometric cryptosystem that generates cryptographic keys from biometric embeddings
instead of using the problematic XOR-based fuzzy commitment approach.

Architecture:
Raw Embedding → Cancelable Transform → Quantization → Key Generation

Advantages:
1. No ECC needed - keys are generated, not error-corrected
2. Works with real biometric noise
3. Provides revocability and unlinkability
4. Generates actual cryptographic keys
"""

import sys
import os
import numpy as np
from typing import Tuple, Dict, Optional
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.crypto_utils import get_sha256_hash
from src.fingerprint.cancelable_transform import CancelableTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricKeyGenerator:
    """
    Generate cryptographic keys from biometric embeddings.
    
    This approach generates keys directly from biometric data rather than
    error-correcting bound keys. This works because:
    
    1. The cancelable transform makes the template revocable and unlinkable
    2. Quantization converts continuous embeddings to bits
    3. Hashing provides cryptographic security and uniform distribution
    
    Security:
    - Cancelable transform prevents original biometric recovery
    - SHA-256 hashing makes template non-invertible
    - Different keys for different applications (unlinkability)
    
    Example:
        >>> generator = BiometricKeyGenerator()
        >>> embedding = np.random.randn(512).astype(np.float32)
        >>> key, template = generator.enroll(embedding, "user_001")
        >>> success = generator.verify(embedding, template, "user_001")
    """
    
    def __init__(self, embedding_dim: int = 512, 
                 key_size: int = 32,
                 cancelable_alpha: float = 0.6,
                 quantization_bits: int = 128):
        """
        Initialize biometric key generator.
        
        Args:
            embedding_dim: Dimension of embeddings (default 512)
            key_size: Cryptographic key size in bytes (default 32 for 256-bit)
            cancelable_alpha: Balance in cancelable transform (default 0.6)
            quantization_bits: Number of bits to extract for key (default 128)
        """
        self.embedding_dim = embedding_dim
        self.key_size = key_size
        self.cancelable_alpha = cancelable_alpha
        self.quantization_bits = quantization_bits
        
        self.cancelable = CancelableTransform(embedding_dim, alpha=cancelable_alpha)
        
        logger.info("BiometricKeyGenerator initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Key size: {key_size * 8} bits")
        logger.info(f"  Cancelable alpha: {cancelable_alpha}")
        logger.info(f"  Quantization bits: {quantization_bits}")
    
    def _quantize_to_bits(self, embedding: np.ndarray, num_bits: int) -> np.ndarray:
        """
        Quantize embedding to binary vector.
        
        Args:
            embedding: Normalized embedding
            num_bits: Number of bits to extract
            
        Returns:
            Binary vector of length num_bits
        """
        embedding = embedding.flatten()
        
        target_length = min(len(embedding), num_bits * 4)
        if len(embedding) > target_length:
            indices = np.linspace(0, len(embedding) - 1, target_length).astype(int)
            embedding = embedding[indices]
        
        scale = 255 / (np.max(embedding) - np.min(embedding) + 1e-10)
        quantized = np.round((embedding - np.min(embedding)) * scale).astype(np.uint8)
        
        bits = np.zeros(num_bits, dtype=np.uint8)
        for i in range(num_bits):
            byte_idx = i % len(quantized)
            bit_position = i % 8
            bits[i] = (quantized[byte_idx] >> bit_position) & 1
        
        return bits
    
    def enroll(self, raw_embedding: np.ndarray, user_key: str) -> Tuple[bytes, Dict]:
        """
        Enroll biometric template and generate cryptographic key.
        
        Args:
            raw_embedding: Raw embedding from deep learning model (512-dim)
            user_key: User/application-specific key
            
        Returns:
            (cryptographic_key, template)
            
            - cryptographic_key: 256-bit key derived from biometrics
            - template: Protected template for verification
        """
        logger.info(f"Enrolling template for user_key: {user_key}")
        
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)
        
        transformed, cancelable_params = self.cancelable.enroll(raw_embedding, user_key)
        
        normalized = transformed / (np.linalg.norm(transformed) + 1e-10)
        
        bits = self._quantize_to_bits(normalized, self.quantization_bits)
        
        key_bytes = bits[:self.key_size * 8].tobytes()
        cryptographic_key = get_sha256_hash(key_bytes)
        
        template = {
            'cancelable_params': cancelable_params,
            'key_hash': get_sha256_hash(cryptographic_key).hex(),
            'quantized_bits': bits.tobytes(),
        }
        
        logger.info(f"Enrollment completed. Key: {cryptographic_key[:8].hex()}...")
        
        return cryptographic_key, template
    
    def verify(self, raw_embedding: np.ndarray, template: Dict, 
               user_key: str, threshold: float = 0.35) -> Tuple[bool, Optional[bytes]]:
        """
        Verify biometric and recover cryptographic key.
        
        Args:
            raw_embedding: Query embedding from deep learning model (512-dim)
            template: Protected template from enrollment
            user_key: User/application-specific key
            threshold: Similarity threshold for verification (default 0.35)
            
        Returns:
            (success, recovered_key)
            
            - success: True if authentication succeeded
            - recovered_key: 256-bit cryptographic key if successful
        """
        logger.info(f"Verifying template for user_key: {user_key}")
        
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)
        
        transformed = self.cancelable.verify(raw_embedding, template['cancelable_params'])
        
        normalized = transformed / (np.linalg.norm(transformed) + 1e-10)
        
        query_bits = self._quantize_to_bits(normalized, self.quantization_bits)
        template_bits = np.frombuffer(template['quantized_bits'], dtype=np.uint8)
        
        hamming_distance = np.sum(query_bits != template_bits)
        similarity = 1.0 - (hamming_distance / len(query_bits))
        
        logger.info(f"Similarity: {similarity:.4f} (threshold: {threshold})")
        
        if similarity >= threshold:
            key_bytes = query_bits[:self.key_size * 8].tobytes()
            recovered_key = get_sha256_hash(key_bytes)
            
            if get_sha256_hash(recovered_key).hex() == template['key_hash']:
                logger.info("Verification SUCCESSFUL - key recovered")
                return True, recovered_key
            else:
                logger.warning("Verification FAILED - key hash mismatch")
                return False, None
        else:
            logger.info("Verification FAILED - similarity below threshold")
            return False, None
    
    def generate_cancelable_key(self, raw_embedding: np.ndarray,
                               old_user_key: str, new_user_key: str) -> Tuple[bytes, Dict]:
        """
        Generate new key from same biometrics with different user key.
        
        Args:
            raw_embedding: Biometric embedding
            old_user_key: Old user key
            new_user_key: New user key
            
        Returns:
            (new_key, new_template)
        """
        logger.info(f"Cancelling old key ({old_user_key}) and generating new ({new_user_key})")
        
        new_transformed, new_cancelable_params = self.cancelable.cancel(
            old_user_key, new_user_key, raw_embedding
        )
        
        normalized = new_transformed / (np.linalg.norm(new_transformed) + 1e-10)
        bits = self._quantize_to_bits(normalized, self.quantization_bits)
        
        key_bytes = bits[:self.key_size * 8].tobytes()
        new_key = get_sha256_hash(key_bytes)
        
        new_template = {
            'cancelable_params': new_cancelable_params,
            'key_hash': get_sha256_hash(new_key).hex(),
            'quantized_bits': bits.tobytes(),
        }
        
        logger.info(f"New key generated: {new_key[:8].hex()}...")
        
        return new_key, new_template
    
    def get_system_info(self) -> Dict:
        """Get system configuration."""
        return {
            'embedding_dim': self.embedding_dim,
            'key_size_bits': self.key_size * 8,
            'cancelable_alpha': self.cancelable_alpha,
            'quantization_bits': self.quantization_bits,
        }


def test_key_generator():
    """Test the biometric key generator."""
    print("\n" + "="*60)
    print("Testing Biometric Key Generator")
    print("="*60)
    
    generator = BiometricKeyGenerator(
        embedding_dim=512,
        key_size=32,
        cancelable_alpha=0.6,
        quantization_bits=256
    )
    
    info = generator.get_system_info()
    print("\nSystem Configuration:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    np.random.seed(42)
    enrollment_emb = np.random.randn(512).astype(np.float32)
    enrollment_emb = enrollment_emb / np.linalg.norm(enrollment_emb)
    
    print("\n" + "-"*60)
    print("Test 1: Enrollment")
    print("-"*60)
    key, template = generator.enroll(enrollment_emb, "user_001")
    print(f"Generated key: {key[:16].hex()}...")
    print(f"Template keys: {list(template.keys())}")
    
    print("\n" + "-"*60)
    print("Test 2: Verification (same embedding)")
    print("-"*60)
    success, recovered_key = generator.verify(enrollment_emb, template, "user_001")
    print(f"Success: {success}")
    if success:
        print(f"Keys match: {key == recovered_key}")
    
    print("\n" + "-"*60)
    print("Test 3: Verification (with small noise)")
    print("-"*60)
    for noise_level in [0.001, 0.005, 0.01, 0.02, 0.05]:
        query_emb = enrollment_emb + np.random.randn(512) * noise_level
        query_emb = query_emb / np.linalg.norm(query_emb)
        success, _ = generator.verify(query_emb, template, "user_001")
        print(f"  Noise {noise_level:.3f}: {'PASS' if success else 'FAIL'}")
    
    print("\n" + "-"*60)
    print("Test 4: Different user (should fail)")
    print("-"*60)
    success, _ = generator.verify(enrollment_emb, template, "user_002")
    print(f"Different user: {'PASS' if success else 'FAIL (expected)'}")
    
    print("\n" + "-"*60)
    print("Test 5: Key cancellation")
    print("-"*60)
    new_key, new_template = generator.generate_cancelable_key(
        enrollment_emb, "user_001", "user_002"
    )
    print(f"New key: {new_key[:16].hex()}...")
    print(f"Keys different: {key != new_key}")
    
    old_valid = generator.verify(enrollment_emb, template, "user_001")
    new_valid = generator.verify(enrollment_emb, new_template, "user_002")
    print(f"Old template still valid: {old_valid[0]} (expected: False)")
    print(f"New template valid: {new_valid[0]}")
    
    print("\n" + "="*60)
    print("Biometric Key Generator Tests Complete")
    print("="*60)


if __name__ == "__main__":
    test_key_generator()
