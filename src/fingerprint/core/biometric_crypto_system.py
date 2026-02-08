import numpy as np
import os
from typing import Tuple, Optional, Dict
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.crypto_utils import get_sha256_hash, xor_bytes, quantize_embedding, dequantize_embedding
from src.fingerprint.ecc_wrapper import ECCWrapper
from src.fingerprint.cancelable_transform import CancelableTransform
from src.fingerprint.fuzzy_commitment import FuzzyCommitment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricCryptoSystem:
    """
    Complete biometric cryptosystem with cancelable transform + fuzzy commitment.
    
    Architecture:
    Raw Embedding → Cancelable Transform → Fuzzy Commitment → Protected Template
    
    Security Properties:
    1. Cancelable: Revocable templates, unlinkable across applications
    2. Fuzzy Commitment: Cryptographically binding, non-invertible
    3. Multi-layered: Defense in depth
    4. Key Generation: Generates 256-bit cryptographic key from biometrics
    
    Template Format:
    {
        'cancelable_params': dict,  # For cancelable layer verification
        'hash_key': str,           # For fuzzy commitment verification
        'helper_data': np.ndarray   # δ from fuzzy commitment
    }
    """
    
    def __init__(self, embedding_dim: int = 512, 
                 key_size: int = 32, 
                 ecc_capacity: float = 0.2,
                 cancelable_alpha: float = 0.6):
        """
        Initialize complete biometric cryptosystem.
        
        Args:
            embedding_dim: Dimension of embeddings (default 512)
            key_size: Cryptographic key size in bytes (default 32 for 256-bit)
            ecc_capacity: Error correction capacity (default 0.2 for 20%)
            cancelable_alpha: Balance in cancelable transform (default 0.6)
        
        Example:
            >>> system = BiometricCryptoSystem(
            ...     embedding_dim=512, 
            ...     key_size=32, 
            ...     ecc_capacity=0.2,
            ...     cancelable_alpha=0.6
            ... )
        """
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.cancelable = CancelableTransform(embedding_dim, alpha=cancelable_alpha)
        self.ecc = ECCWrapper(message_size=key_size, error_capacity_percent=ecc_capacity)
        self.fuzzy = FuzzyCommitment(self.ecc)
        
        logger.info("="*60)
        logger.info("Biometric CryptoSystem initialized")
        logger.info("="*60)
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Key size: {key_size * 8} bits")
        logger.info(f"  ECC capacity: {ecc_capacity * 100:.1f}%")
        logger.info(f"  Cancelable alpha: {cancelable_alpha}")
        logger.info("="*60)
    
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
            >>> embedding = backbone(img)  # Get embedding from trained model
            >>> template = system.enroll(embedding, "user_001")
            >>> print(template.keys())
            >>> # dict_keys(['cancelable_params', 'hash_key', 'helper_data'])
        """
        logger.info(f"Enrolling template for user_key: {user_key}")
        
        # Ensure numpy array and correct shape
        if TORCH_AVAILABLE and hasattr(raw_embedding, 'detach'):
            raw_embedding = raw_embedding.detach().cpu().numpy()
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)
        
        # Step 1: Apply cancelable transform
        logger.debug("Step 1: Applying cancelable transform...")
        transformed_embedding, cancelable_params = self.cancelable.enroll(
            raw_embedding, user_key
        )
        
        # Step 2: Apply fuzzy commitment
        logger.debug("Step 2: Applying fuzzy commitment...")
        hash_key, helper_data = self.fuzzy.enroll(transformed_embedding)
        
        # Step 3: Construct protected template
        template = {
            'cancelable_params': cancelable_params,
            'hash_key': hash_key,
            'helper_data': helper_data,
        }
        
        logger.info(f"Enrollment completed. Template size: {len(helper_data)} bytes")
        
        return template
    
    def verify(self, raw_embedding: np.ndarray, template: Dict, 
               user_key: str) -> Tuple[bool, Optional[bytes]]:
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
        
        Example:
            >>> system = BiometricCryptoSystem()
            >>> template = system.enroll(enrollment_embedding, "user_001")
            >>> # Later, during authentication...
            >>> query_embedding = backbone(query_img)
            >>> success, key = system.verify(query_embedding, template, "user_001")
            >>> print(f"Authenticated: {success}")
            >>> if success:
            ...     print(f"Recovered key: {key.hex()}")
        """
        logger.info(f"Verifying template for user_key: {user_key}")
        
        # Ensure numpy array and correct shape
        if TORCH_AVAILABLE and hasattr(raw_embedding, 'detach'):
            raw_embedding = raw_embedding.detach().cpu().numpy()
        raw_embedding = np.array(raw_embedding).flatten().astype(np.float32)
        
        # Step 1: Apply cancelable transform
        logger.debug("Step 1: Applying cancelable transform...")
        transformed_embedding = self.cancelable.verify(
            raw_embedding, template['cancelable_params']
        )
        
        # Step 2: Apply fuzzy commitment verification
        logger.debug("Step 2: Verifying fuzzy commitment...")
        success, recovered_key = self.fuzzy.verify(
            transformed_embedding,
            template['hash_key'],
            template['helper_data']
        )
        
        if success:
            logger.info("Verification SUCCESSFUL")
        else:
            logger.info("Verification FAILED")
        
        return success, recovered_key
    
    def cancel_and_reissue(self, raw_embedding: np.ndarray, 
                          old_user_key: str, new_user_key: str) -> Dict:
        """
        Cancel old template and issue new one with different user key.
        
        Args:
            raw_embedding: Biometric embedding (can use enrollment sample)
            old_user_key: Old user key to cancel
            new_user_key: New user key to issue
        
        Returns:
            New protected template dictionary
        
        Example:
            >>> system = BiometricCryptoSystem()
            >>> # User wants to cancel old template
            >>> new_template = system.cancel_and_reissue(
            ...     enrollment_embedding, 
            ...     "user_001", 
            ...     "user_002"
            ... )
            >>> # Old templates are now invalid
        """
        logger.info(f"Cancelling old template ({old_user_key}) and issuing new ({new_user_key})")
        
        # Cancel old template and get new transformed embedding
        new_transformed, new_cancelable_params = self.cancelable.cancel(
            old_user_key, new_user_key, raw_embedding
        )
        
        # Re-enroll with fuzzy commitment
        hash_key, helper_data = self.fuzzy.enroll(new_transformed)
        
        # Construct new template
        new_template = {
            'cancelable_params': new_cancelable_params,
            'hash_key': hash_key,
            'helper_data': helper_data,
        }
        
        logger.info("New template issued successfully")
        
        return new_template
    
    def get_system_info(self) -> Dict:
        """
        Get system configuration and parameters.
        
        Returns:
            Dictionary with system configuration
        """
        ecc_symbols = len(self.ecc.rsc.encode(b'\x00' * self.ecc.message_size)) - self.ecc.message_size
        
        return {
            'embedding_dim': self.embedding_dim,
            'key_size_bits': self.ecc.message_size * 8,
            'key_size_bytes': self.ecc.message_size,
            'ecc_capacity': self.ecc.message_size * 0.2 / (self.ecc.message_size + ecc_symbols),  # Approximate
            'ecc_symbols': ecc_symbols,
            'codeword_length': self.ecc.message_size + ecc_symbols,
            'cancelable_alpha': self.cancelable.alpha,
        }


if __name__ == "__main__":
    # Test complete biometric crypto system
    print("\n" + "="*60)
    print("Testing Complete Biometric CryptoSystem")
    print("="*60)
    
    system = BiometricCryptoSystem(
        embedding_dim=512,
        key_size=32,
        ecc_capacity=0.2,
        cancelable_alpha=0.6
    )
    
    # Print system info
    print("\nSystem Configuration:")
    info = system.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Generate test embeddings
    print("\nGenerating test embeddings...")
    enrollment_emb = np.random.randn(512).astype(np.float32)
    enrollment_emb = enrollment_emb / np.linalg.norm(enrollment_emb)
    
    # Test 1: Enrollment
    print("\n" + "-"*60)
    print("Test 1: Enrollment")
    print("-"*60)
    template = system.enroll(enrollment_emb, "user_001")
    print(f"Template keys: {template.keys()}")
    print(f"Cancel params: {template['cancelable_params']['user_key']}")
    print(f"Key hash: {template['hash_key'][:16]}...")
    print(f"Helper data size: {len(template['helper_data'])} bytes")
    
    # Test 2: Verification with same embedding
    print("\n" + "-"*60)
    print("Test 2: Verification (same embedding)")
    print("-"*60)
    success, key = system.verify(enrollment_emb, template, "user_001")
    print(f"Success: {success}")
    if success:
        print(f"Recovered key: {key.hex()[:32] if key else 'None'}...")
    
    # Test 3: Verification with small noise
    print("\n" + "-"*60)
    print("Test 3: Verification (small noise)")
    print("-"*60)
    query_emb = enrollment_emb + np.random.randn(512) * 0.001
    query_emb = query_emb / np.linalg.norm(query_emb)
    success, key = system.verify(query_emb, template, "user_001")
    print(f"Success: {success}")
    
    # Test 4: Cancellation and reissuance
    print("\n" + "-"*60)
    print("Test 4: Cancellation and Reissuance")
    print("-"*60)
    new_template = system.cancel_and_reissue(
        enrollment_emb, "user_001", "user_002"
    )
    print(f"New template user_key: {new_template['cancelable_params']['user_key']}")
    
    # Old template should no longer work
    print("\nTesting old template (should fail):")
    success, _ = system.verify(query_emb, template, "user_001")
    print(f"Old template valid: {success}")
    
    # New template should work
    print("\nTesting new template (should succeed):")
    success, key = system.verify(query_emb, new_template, "user_002")
    print(f"New template valid: {success}")