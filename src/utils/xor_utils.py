def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    Performs a byte-wise XOR of two byte strings.

    Args:
        a (bytes): The first byte string.
        b (bytes): The second byte string.

    Returns:
        bytes: The result of XORing a and b.

    Raises:
        ValueError: If the lengths of a and b are different.
    """
    if len(a) != len(b):
        raise ValueError(f"Input bytes must have the same length. Got {len(a)} and {len(b)}.")
    
    return bytes(x ^ y for x, y in zip(a, b))
