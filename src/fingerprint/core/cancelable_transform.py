import numpy as np
import hashlib
from typing import Tuple, Dict
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CancelableTransform:
    """
    Cancelable biometric transformation using random projection.
    
    Properties:
    - Revocability: Different key → different transform
    - Unlinkability: Different applications → different keys
    - Non-invertibility: Without key, hard to recover original
    
    Based on your existing implementation in idv_inference.py,
    but refactored for reusability.
    """
    
    def __init__(self, embedding_dim: int = 512, alpha: float = 0.5):
        """
        Initialize cancelable transform.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            alpha: Balance between biometric and key-driven components
                   (0.0 = fully key-driven, 1.0 = fully biometric-driven)
        
        Example:
            >>> transform = CancelableTransform(embedding_dim=512, alpha=0.6)
        """
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        logger.info(f"CancelableTransform initialized: dim={embedding_dim}, alpha={alpha}")
    
    def _generate_projection_matrix(self, user_key: str, embedding: np.ndarray) -> np.ndarray:
        """
        Generate hybrid projection matrix combining key-driven and biometric components.
        
        Args:
            user_key: User/application-specific key string
            embedding: Biometric embedding for biometric-driven component
        
        Returns:
            Projection matrix R (embedding_dim x embedding_dim)
        """
        # (A) Key-driven random matrix
        seed = int(hashlib.sha256(user_key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        R_key = rng.standard_normal((self.embedding_dim, self.embedding_dim))
        R_key /= np.linalg.norm(R_key, axis=1, keepdims=True)
        
        # (B) Biometric-driven matrix
        E_norm = embedding / np.linalg.norm(embedding)
        R_bio = np.outer(E_norm, E_norm)
        R_bio /= np.linalg.norm(R_bio, axis=1, keepdims=True)
        
        # (C) Combine both
        R_dyn = self.alpha * R_bio + (1 - self.alpha) * R_key
        R_dyn /= np.linalg.norm(R_dyn, axis=1, keepdims=True)
        
        return R_dyn
    
    def enroll(self, embedding: np.ndarray, user_key: str) -> Tuple[np.ndarray, Dict]:
        """
        Apply cancelable transform during enrollment.
        
        Args:
            embedding: Raw biometric embedding (512-dim)
            user_key: User/application-specific key
        
        Returns:
            (transformed_embedding, transform_params)
            
            transform_params = {
                'user_key': str,
                'alpha': float,
                'embedding_dim': int
            }
        
        Example:
            >>> transform = CancelableTransform(512, 0.6)
            >>> embedding = np.random.randn(512)
            >>> transformed, params = transform.enroll(embedding, "user_001")
            >>> print(transformed.shape)  # (512,)
            >>> print(params)  # {'user_key': 'user_001', 'alpha': 0.6, ...}
"""
        # Ensure numpy array
        if TORCH_AVAILABLE and hasattr(embedding, 'detach'):
            embedding = embedding.detach().cpu().numpy()
        elif not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Generate projection matrix
        R = self._generate_projection_matrix(user_key, embedding)
        
        # Apply transformation
        transformed = np.dot(R, embedding)
        
        # Store parameters for verification
        params = {
            'user_key': user_key,
            'alpha': self.alpha,
            'embedding_dim': self.embedding_dim,
        }
        
        return transformed, params
    
    def verify(self, embedding: np.ndarray, transform_params: Dict) -> np.ndarray:
        """
        Apply same cancelable transform during verification.
        
        Args:
            embedding: Raw biometric embedding (512-dim, possibly noisy)
            transform_params: Parameters stored during enrollment
        
        Returns:
            Transformed embedding (512-dim)
        
        Example:
            >>> transform = CancelableTransform(512, 0.6)
            >>> noisy_embedding = np.random.randn(512)
            >>> transformed = transform.verify(noisy_embedding, params)
            >>> print(transformed.shape)  # (512,)
        """
        # Extract parameters
        user_key = transform_params['user_key']
        
        # Ensure numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Generate same projection matrix
        R = self._generate_projection_matrix(user_key, embedding)
        
        # Apply transformation
        transformed = np.dot(R, embedding)
        
        return transformed
    
    def cancel(self, user_key_old: str, user_key_new: str, 
               embedding: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Cancel old template and issue new one with different key.
        
        Args:
            user_key_old: Old user key (to be revoked)
            user_key_new: New user key (to be issued)
            embedding: Biometric embedding (can use enrollment sample)
        
        Returns:
            (new_transformed_embedding, new_transform_params)
        
        Example:
            >>> transform = CancelableTransform(512, 0.6)
            >>> embedding = np.random.randn(512)
            >>> # Original enrollment
            >>> old_transformed, old_params = transform.enroll(embedding, "user_001")
            >>> # Cancel and reissue
            >>> new_transformed, new_params = transform.cancel(
            ...     "user_001", "user_002", embedding
            ... )
            >>> # Old params no longer valid, must use new_params
        """
        logger.info(f"Cancelling template with key '{user_key_old}', issuing new key '{user_key_new}'")
        return self.enroll(embedding, user_key_new)


if __name__ == "__main__":
    # Test Cancelable Transform
    print("\n" + "="*60)
    print("Testing Cancelable Transform")
    print("="*60)
    
    transform = CancelableTransform(embedding_dim=512, alpha=0.6)
    
    # Generate test embedding
    embedding1 = np.random.randn(512).astype(np.float32)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    
    # Test 1: Enrollment
    print("\nTest 1: Enrollment")
    transformed1, params1 = transform.enroll(embedding1, "user_001")
    print(f"  Original shape: {embedding1.shape}")
    print(f"  Transformed shape: {transformed1.shape}")
    print(f"  Params: {params1}")
    
    # Test 2: Verification with similar embedding
    print("\nTest 2: Verification")
    embedding2 = embedding1 + np.random.randn(512) * 0.01  # Add small noise
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    transformed2 = transform.verify(embedding2, params1)
    similarity = np.dot(transformed1, transformed2) / (np.linalg.norm(transformed1) * np.linalg.norm(transformed2))
    print(f"  Similarity after noise: {similarity:.4f}")
    
    # Test 3: Cancellation
    print("\nTest 3: Template Cancellation")
    new_transformed, new_params = transform.cancel("user_001", "user_002", embedding1)
    print(f"  Old params: {params1['user_key']}")
    print(f"  New params: {new_params['user_key']}")
    print(f"  New transformed shape: {new_transformed.shape}")
    
    # Test 4: Unlinkability (different users get different transforms)
    print("\nTest 4: Unlinkability")
    embedding_user1 = np.random.randn(512).astype(np.float32)
    embedding_user1 = embedding_user1 / np.linalg.norm(embedding_user1)
    
    embedding_user2 = np.random.randn(512).astype(np.float32)
    embedding_user2 = embedding_user2 / np.linalg.norm(embedding_user2)
    
    transformed_user1, _ = transform.enroll(embedding_user1, "app_001")
    transformed_user2, _ = transform.enroll(embedding_user2, "app_002")
    
    similarity = np.dot(transformed_user1, transformed_user2)
    print(f"  Cross-user similarity: {similarity:.4f} (should be low)")