from typing import List

def quantize_embedding(embedding: List[float], min_val: float = -1.0, max_val: float = 1.0) -> bytes:
    """
    Quantizes a list of floats into bytes (8 bits per dimension).
    
    The mapping is linear from [min_val, max_val] to [0, 255].
    Values outside [min_val, max_val] are clamped.

    Args:
        embedding (List[float]): The input embedding (usually 512-dim).
        min_val (float): The minimum expected value in the embedding.
        max_val (float): The maximum expected value in the embedding.

    Returns:
        bytes: The quantized embedding as a byte string.
    """
    scale = 255.0 / (max_val - min_val)
    quantized_values = []
    
    for val in embedding:
        # Clamp value
        val = max(min_val, min(val, max_val))
        # Scale and round
        int_val = int(round((val - min_val) * scale))
        quantized_values.append(int_val)
        
    return bytes(quantized_values)

def dequantize_embedding(quantized: bytes, min_val: float = -1.0, max_val: float = 1.0) -> List[float]:
    """
    Reconstructs a list of floats from a quantized byte string.

    Args:
        quantized (bytes): The quantized embedding.
        min_val (float): The minimum value used during quantization.
        max_val (float): The maximum value used during quantization.

    Returns:
        List[float]: The reconstructed embedding.
    """
    scale = (max_val - min_val) / 255.0
    return [byte * scale + min_val for byte in quantized]
