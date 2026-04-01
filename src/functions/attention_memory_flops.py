import numpy as np


def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    #Each element of q is multiplied and added to k.
    qk_flops = B * h * 2 * (N*d) * N
    
    #Each element of qk is subjected to finding max, subtract max, exponentiate, sum, divide
    softmax_flops = B * h * 5 * N**2

    # Each element of qk is mutiplied and added to v.
    pv_flops = B * h* 2 * N**2 * d

    total_flops = B * h * N**2 * (5 + 4*d)

    # Memory count (including write and reads)
    memory_qk =  B * h * ( 2 * (N * d) + N ** 2)
    memory_softmax = 2 * B * h * (N**2) 
    memory_pv = B * h * (N**2 + (N*d) + (N*d)) 
    memory_read_writes = memory_qk + memory_softmax + memory_pv
    memory_bytes = memory_read_writes * bytes_per_element

    intensity = total_flops / memory_bytes
    arithmetic_intensity = np.round(intensity,2)

    return {
        'qk_flops': qk_flops,
        'softmax_flops': softmax_flops,
        'pv_flops': pv_flops,
        'total_flops': total_flops,
        'memory_bytes': memory_bytes,
        'arithmetic_intensity': arithmetic_intensity,
    }