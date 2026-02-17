import hashlib

def get_sha256_hash(data: bytes) -> bytes:
    """
    Returns the SHA-256 hash of the input bytes.

    Args:
        data (bytes): Input data to hash.

    Returns:
        bytes: The 32-byte SHA-256 hash.
    """
    return hashlib.sha256(data).digest()
