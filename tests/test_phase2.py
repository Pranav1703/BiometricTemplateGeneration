import pytest
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.fingerprint.core.cancelable_transform import CancelableTransform
from src.fingerprint.core.fuzzy_commitment import FuzzyCommitment
from src.utils.ecc_utils import ECCWrapper
from src.fingerprint.core.biometric_crypto_system import BiometricCryptoSystem


class TestCancelableTransform:
    """Test suite for CancelableTransform class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.transform = CancelableTransform(embedding_dim=512, alpha=0.6)
        self.embedding = np.random.randn(512).astype(np.float32)
        self.embedding = self.embedding / np.linalg.norm(self.embedding)
        self.user_key = "test_user_001"

    def test_enrollment(self):
        """Test enrollment process."""
        transformed, params = self.transform.enroll(self.embedding, self.user_key)

        # Check shapes
        assert transformed.shape == (512,)
        assert isinstance(params, dict)
        assert params["user_key"] == self.user_key
        assert params["alpha"] == 0.6
        assert params["embedding_dim"] == 512

    def test_verification_same_embedding(self):
        """Test verification with same embedding."""
        transformed, params = self.transform.enroll(self.embedding, self.user_key)
        verified = self.transform.verify(self.embedding, params)

        # Should be very similar (allowing for numerical precision)
        similarity = np.dot(transformed, verified) / (
            np.linalg.norm(transformed) * np.linalg.norm(verified)
        )
        assert similarity > 0.999

    def test_verification_noisy_embedding(self):
        """Test verification with noisy embedding."""
        transformed, params = self.transform.enroll(self.embedding, self.user_key)

        # Add small noise
        noisy_embedding = self.embedding + np.random.randn(512) * 0.01
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)

        verified = self.transform.verify(noisy_embedding, params)

        # Should still be reasonably similar
        similarity = np.dot(transformed, verified) / (
            np.linalg.norm(transformed) * np.linalg.norm(verified)
        )
        assert similarity > 0.95

    def test_deterministic_same_key(self):
        """Test that same key produces same transform."""
        transformed1, params1 = self.transform.enroll(self.embedding, self.user_key)
        transformed2, params2 = self.transform.enroll(self.embedding, self.user_key)

        # Should be identical
        np.testing.assert_array_almost_equal(transformed1, transformed2)
        assert params1["user_key"] == params2["user_key"]

    def test_different_keys_different_transforms(self):
        """Test that different keys produce different transforms."""
        key1 = "user_001"
        key2 = "user_002"

        transformed1, params1 = self.transform.enroll(self.embedding, key1)
        transformed2, params2 = self.transform.enroll(self.embedding, key2)

        # Should be different
        similarity = np.dot(transformed1, transformed2) / (
            np.linalg.norm(transformed1) * np.linalg.norm(transformed2)
        )
        assert similarity < 0.9  # Should be quite different
        assert params1["user_key"] != params2["user_key"]

    def test_cancellation(self):
        """Test template cancellation and reissuance."""
        old_key = "old_user_key"
        new_key = "new_user_key"

        # Original enrollment
        old_transformed, old_params = self.transform.enroll(self.embedding, old_key)

        # Cancel and reissue
        new_transformed, new_params = self.transform.cancel(
            old_key, new_key, self.embedding
        )

        # Check new params
        assert new_params["user_key"] == new_key
        assert new_params["user_key"] != old_params["user_key"]

        # New transform should work with new key
        verified_new = self.transform.verify(self.embedding, new_params)
        similarity_new = np.dot(new_transformed, verified_new) / (
            np.linalg.norm(new_transformed) * np.linalg.norm(verified_new)
        )
        assert similarity_new > 0.999

    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        if TORCH_AVAILABLE:
            torch_embedding = torch.from_numpy(self.embedding)

            transformed, params = self.transform.enroll(torch_embedding, self.user_key)
            verified = self.transform.verify(torch_embedding, params)

            # Should work the same as numpy arrays
            similarity = np.dot(transformed, verified) / (
                np.linalg.norm(transformed) * np.linalg.norm(verified)
            )
            assert similarity > 0.999
        else:
            pytest.skip("torch not available")


class TestFuzzyCommitment:
    """Test suite for FuzzyCommitment class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fuzzy = FuzzyCommitment(key_size=32, ecc_capacity=0.2, embedding_dim=512)
        self.embedding = np.random.randn(512).astype(np.float32)
        self.embedding = self.embedding / np.linalg.norm(self.embedding)

    def test_enrollment(self):
        """Test enrollment process."""
        hash_key, helper_data = self.fuzzy.enroll(self.embedding)

        # Check types and sizes
        assert isinstance(hash_key, str)
        assert len(hash_key) == 64  # SHA-256 hex string
        assert isinstance(helper_data, (np.ndarray, bytes))
        assert len(helper_data) == 512  # Should match embedding size

    def test_verification_same_embedding(self):
        """Test verification with same embedding."""
        hash_key, helper_data = self.fuzzy.enroll(self.embedding)
        success, recovered_key = self.fuzzy.verify(
            self.embedding, hash_key, helper_data
        )

        assert success is True
        assert recovered_key is not None
        assert len(recovered_key) == 32  # 256-bit key

    def test_verification_small_noise(self):
        """Test verification with small noise."""
        hash_key, helper_data = self.fuzzy.enroll(self.embedding)

        # Add very small noise
        noisy_embedding = self.embedding + np.random.randn(512) * 0.001
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)

        success, recovered_key = self.fuzzy.verify(
            noisy_embedding, hash_key, helper_data
        )

        # Should still succeed with very small noise
        assert success is True

    def test_verification_large_noise(self):
        """Test verification with large noise (should fail)."""
        hash_key, helper_data = self.fuzzy.enroll(self.embedding)

        # Add large noise
        noisy_embedding = self.embedding + np.random.randn(512) * 0.1
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)

        success, recovered_key = self.fuzzy.verify(
            noisy_embedding, hash_key, helper_data
        )

        # Should fail with large noise
        assert success is False
        assert recovered_key is None

    def test_key_recovery_consistency(self):
        """Test that recovered key is consistent across successful verifications."""
        hash_key, helper_data = self.fuzzy.enroll(self.embedding)

        # Verify multiple times with same embedding
        success1, key1 = self.fuzzy.verify(self.embedding, hash_key, helper_data)
        success2, key2 = self.fuzzy.verify(self.embedding, hash_key, helper_data)

        assert success1 is True
        assert success2 is True
        assert key1 == key2  # Should be identical

    def test_different_embeddings_different_keys(self):
        """Test that different embeddings generate different keys."""
        embedding2 = np.random.randn(512).astype(np.float32)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        hash_key1, helper_data1 = self.fuzzy.enroll(self.embedding)
        hash_key2, helper_data2 = self.fuzzy.enroll(embedding2)

        # Keys should be different
        assert hash_key1 != hash_key2
        assert not np.array_equal(helper_data1, helper_data2)


class TestBiometricCryptoSystem:
    """Test suite for BiometricCryptoSystem class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.system = BiometricCryptoSystem(
            embedding_dim=512, key_size=32, ecc_capacity=0.2, cancelable_alpha=0.6
        )
        self.embedding = np.random.randn(512).astype(np.float32)
        self.embedding = self.embedding / np.linalg.norm(self.embedding)
        self.user_key = "test_user_001"

    def test_enrollment(self):
        """Test enrollment process."""
        template = self.system.enroll(self.embedding, self.user_key)

        # Check template structure
        assert isinstance(template, dict)
        assert "cancelable_params" in template
        assert "hash_key" in template
        assert "helper_data" in template

        # Check cancelable params
        assert template["cancelable_params"]["user_key"] == self.user_key
        assert template["cancelable_params"]["alpha"] == 0.6
        assert template["cancelable_params"]["embedding_dim"] == 512

        # Check fuzzy commitment parts
        assert isinstance(template["hash_key"], str)
        assert len(template["hash_key"]) == 64  # SHA-256 hex
        assert isinstance(template["helper_data"], np.ndarray)
        assert len(template["helper_data"]) == 512

    def test_verification_same_embedding(self):
        """Test verification with same embedding."""
        template = self.system.enroll(self.embedding, self.user_key)
        success, recovered_key = self.system.verify(
            self.embedding, template, self.user_key
        )

        assert success is True
        assert recovered_key is not None
        assert len(recovered_key) == 32  # 256-bit key

    def test_verification_noisy_embedding(self):
        """Test verification with noisy embedding."""
        template = self.system.enroll(self.embedding, self.user_key)

        # Add small noise
        noisy_embedding = self.embedding + np.random.randn(512) * 0.001
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)

        success, recovered_key = self.system.verify(
            noisy_embedding, template, self.user_key
        )

        # Should still succeed with small noise
        assert success is True

    def test_verification_wrong_user_key(self):
        """Test verification with wrong user key (should fail)."""
        template = self.system.enroll(self.embedding, self.user_key)
        wrong_key = "wrong_user_key"

        success, recovered_key = self.system.verify(self.embedding, template, wrong_key)

        # Should fail with wrong key
        assert success is False
        assert recovered_key is None

    def test_cancellation_and_reissuance(self):
        """Test template cancellation and reissuance."""
        old_key = "old_user_key"
        new_key = "new_user_key"

        # Original enrollment
        old_template = self.system.enroll(self.embedding, old_key)

        # Cancel and reissue
        new_template = self.system.cancel_and_reissue(self.embedding, old_key, new_key)

        # Check new template
        assert new_template["cancelable_params"]["user_key"] == new_key
        assert (
            new_template["cancelable_params"]["user_key"]
            != old_template["cancelable_params"]["user_key"]
        )

        # Old template should fail
        success_old, _ = self.system.verify(self.embedding, old_template, old_key)
        assert success_old is False

        # New template should succeed
        success_new, recovered_key = self.system.verify(
            self.embedding, new_template, new_key
        )
        assert success_new is True
        assert recovered_key is not None

    def test_system_info(self):
        """Test system information retrieval."""
        info = self.system.get_system_info()

        # Check required fields
        assert "embedding_dim" in info
        assert "key_size_bits" in info
        assert "key_size_bytes" in info
        assert "ecc_capacity" in info
        assert "ecc_symbols" in info
        assert "codeword_length" in info
        assert "cancelable_alpha" in info

        # Check values
        assert info["embedding_dim"] == 512
        assert info["key_size_bits"] == 256
        assert info["key_size_bytes"] == 32
        assert info["cancelable_alpha"] == 0.6

    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        if TORCH_AVAILABLE:
            torch_embedding = torch.from_numpy(self.embedding)

            template = self.system.enroll(torch_embedding, self.user_key)
            success, recovered_key = self.system.verify(
                torch_embedding, template, self.user_key
            )

            # Should work the same as numpy arrays
            assert success is True
            assert recovered_key is not None
        else:
            pytest.skip("torch not available")


if __name__ == "__main__":
    # Run tests manually
    print("Running Phase 2 tests...")

    # Test CancelableTransform
    print("\nTesting CancelableTransform...")
    ct = TestCancelableTransform()
    ct.setup_method()
    ct.test_enrollment()
    ct.test_verification_same_embedding()
    ct.test_verification_noisy_embedding()
    ct.test_deterministic_same_key()
    ct.test_different_keys_different_transforms()
    ct.test_cancellation()
    ct.test_torch_tensor_input()
    print("✓ CancelableTransform tests passed")

    # Test FuzzyCommitment
    print("\nTesting FuzzyCommitment...")
    fc = TestFuzzyCommitment()
    fc.setup_method()
    fc.test_enrollment()
    fc.test_verification_same_embedding()
    fc.test_verification_small_noise()
    fc.test_verification_large_noise()
    fc.test_key_recovery_consistency()
    fc.test_different_embeddings_different_keys()
    print("✓ FuzzyCommitment tests passed")

    # Test BiometricCryptoSystem
    print("\nTesting BiometricCryptoSystem...")
    bcs = TestBiometricCryptoSystem()
    bcs.setup_method()
    bcs.test_enrollment()
    bcs.test_verification_same_embedding()
    bcs.test_verification_noisy_embedding()
    bcs.test_verification_wrong_user_key()
    bcs.test_cancellation_and_reissuance()
    bcs.test_system_info()
    bcs.test_torch_tensor_input()
    print("✓ BiometricCryptoSystem tests passed")

    print("\n🎉 All Phase 2 tests passed!")
