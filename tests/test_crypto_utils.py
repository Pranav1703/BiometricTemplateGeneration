
import pytest
import numpy as np
import random
from reedsolo import ReedSolomonError
from src.crypto_utils.hash_utils import get_sha256_hash
from src.crypto_utils.xor_utils import xor_bytes
from src.crypto_utils.quantization import quantize_embedding, dequantize_embedding
from src.crypto_utils.ecc_utils import ECCWrapper

def test_sha256_determinism():
    data = b"test_data"
    hash1 = get_sha256_hash(data)
    hash2 = get_sha256_hash(data)
    assert hash1 == hash2
    assert len(hash1) == 32

def test_xor_correctness():
    a = b'\x00\xFF\xAA'
    b = b'\xFF\x00\x55'
    expected = b'\xFF\xFF\xFF'
    assert xor_bytes(a, b) == expected
    
    with pytest.raises(ValueError):
        xor_bytes(a, b'\x00')

def test_quantization_accuracy():
    # Generate random embedding
    dims = 512
    # embeddings usually normalized, so range -1 to 1 is typical
    original = [random.uniform(-1.0, 1.0) for _ in range(dims)]
    
    quantized = quantize_embedding(original)
    assert len(quantized) == dims
    
    reconstructed = dequantize_embedding(quantized)
    
    # Calculate MSE
    mse = np.mean((np.array(original) - np.array(reconstructed)) ** 2)
    print(f"Quantization MSE: {mse}")
    assert mse < 0.01

def test_ecc_correction():
    wrapper = ECCWrapper(message_size=32, error_capacity_percent=0.2)
    original_msg = b"A" * 32
    encoded = wrapper.encode(original_msg)
    
    # Introduce errors
    # 20% of 32 is ~6 bytes. The wrapper likely sets up RS to correct ~6 bytes.
    # Let's try corrupting 6 bytes.
    corrupted = bytearray(encoded)
    for i in range(6):
        corrupted[i] ^= 0xFF # Flip bits
    
    decoded = wrapper.decode(corrupted)
    assert decoded == original_msg

def test_ecc_failure_beyond_capacity():
    wrapper = ECCWrapper(message_size=32, error_capacity_percent=0.2)
    original_msg = b"B" * 32
    encoded = wrapper.encode(original_msg)
    
    # Introduce too many errors
    # If capacity is ~6, 12 errors should definitely fail or produce wrong result
    # Reed-Solomon will either raise error or return wrong data if overwhelmed (but usually raises if it detects)
    # With enough parity, it can detect uncorrectable errors.
    
    corrupted = bytearray(encoded)
    # Corrupt 20 bytes
    for i in range(20):
        corrupted[i] ^= 0xFF
        
    with pytest.raises(ReedSolomonError):
        wrapper.decode(corrupted)

