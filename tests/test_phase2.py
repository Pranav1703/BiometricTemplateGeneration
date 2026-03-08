import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.fingerprint.core.cancelable_transform import CancelableTransform
from src.fingerprint.core.biometric_crypto_system import BiometricCryptoSystem


class TestCancelableTransform:
    """Test suite for CancelableTransform class."""

    def test_enrollment(self):
        """Test enrollment process."""
        np.random.seed(42)
        transform = CancelableTransform(embedding_dim=512, alpha=0.6)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        transformed, params = transform.enroll(embedding, "test")

        assert transformed.shape == (512,)
        assert params["user_key"] == "test"
        assert params["alpha"] == 0.6

    def test_verification(self):
        """Test verification."""
        np.random.seed(100)
        transform = CancelableTransform(embedding_dim=512, alpha=0.6)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        transformed, params = transform.enroll(embedding, "test")
        verified = transform.verify(embedding, params)

        similarity = np.dot(transformed, verified) / (
            np.linalg.norm(transformed) * np.linalg.norm(verified)
        )
        assert similarity > 0.999


class TestBiometricCryptoSystem:
    """Test suite for BiometricCryptoSystem class."""

    def test_enrollment(self):
        """Test enrollment."""
        np.random.seed(200)
        system = BiometricCryptoSystem(
            embedding_dim=512, key_size=32, cancelable_alpha=0.6
        )
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        template = system.enroll(embedding, "test")

        assert "transformed" in template
        assert "salt" in template
        assert "key_hash" in template

    def test_system_info(self):
        """Test system info."""
        np.random.seed(300)
        system = BiometricCryptoSystem(
            embedding_dim=512, key_size=32, cancelable_alpha=0.6
        )

        info = system.get_system_info()

        assert info["method"] == "Cancelable + PBKDF2"


def run_phase2_tests():
    """Run all Phase 2 tests manually."""
    print("Running Phase 2 tests...")

    print("\nTesting CancelableTransform...")
    ct = TestCancelableTransform()
    ct.test_enrollment()
    ct.test_verification()
    print("OK: CancelableTransform tests passed")

    print("\nTesting BiometricCryptoSystem...")
    bcs = TestBiometricCryptoSystem()
    bcs.test_enrollment()
    bcs.test_system_info()
    print("OK: BiometricCryptoSystem tests passed")

    print("\nAll Phase 2 tests passed!")


if __name__ == "__main__":
    run_phase2_tests()
