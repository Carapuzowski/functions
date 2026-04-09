import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    num_blocks = tensor.shape[0]//block_size
    block_tensor = np.reshape(tensor, (num_blocks, block_size))

    block_max = np.max(np.abs(block_tensor), axis = 1)

    scales = np.maximum(block_max / 448.0, 1e-12)
    
    quantized = block_tensor / (scales[:, np.newaxis] + 1e-12)
    quantized = np.round(quantized)
    quantized = np.clip(quantized, -448, 448).flatten()
    return (quantized, scales)


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    num_blocks = quantized.shape[0] // block_size
    quantized_blocks = quantized.reshape(num_blocks, block_size)
    
    dequantized = quantized_blocks * scales[:, np.newaxis]
    
    return dequantized.flatten()