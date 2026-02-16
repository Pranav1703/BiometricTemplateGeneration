import numpy as np

def quantize_embedding(embedding, bits=8, min_val=None, max_val=None):
    """Quantizes a list of floats into bytes."""
    embedding = np.array(embedding, dtype=np.float32)
    
    if min_val is None:
        min_val = np.min(embedding)
    if max_val is None:
        max_val = np.max(embedding)
    
    scale = (2 ** bits - 1) / (max_val - min_val + 1e-10)
    quantized_values = np.clip(np.round((embedding - min_val) * scale), 0, 2 ** bits - 1)
    
    return quantized_values.astype(np.uint8).tobytes()


def dequantize_embedding(quantized, bits=8, min_val=None, max_val=None):
    """Reconstructs a list of floats from quantized bytes."""
    quantized_arr = np.frombuffer(quantized, dtype=np.uint8)
    
    if min_val is None:
        min_val = 0
    if max_val is None:
        max_val = 1.0
    
    scale = (max_val - min_val) / (2 ** bits - 1)
    reconstructed = (quantized_arr.astype(np.float32) * scale + min_val).tolist()
    
    return reconstructed


def robust_quantize(embedding, num_bits=12):
    """Robust quantization for biometric embeddings."""
    embedding = embedding.astype(np.float32)
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    scale = (2 ** num_bits - 1)
    quantized = np.clip(np.round(embedding * scale + scale / 2), 0, 2 * scale)
    quantized = (quantized / 2).astype(np.uint8)
    
    return quantized.tobytes()


def robust_dequantize(quantized, num_bits=12):
    """Robust dequantization for biometric embeddings."""
    quantized_arr = np.frombuffer(quantized, dtype=np.uint8).astype(np.float32)
    
    scale = (2 ** num_bits - 1)
    reconstructed = (quantized_arr - scale / 2) / scale
    
    return reconstructed
