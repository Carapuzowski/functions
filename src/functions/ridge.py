import numpy as np

def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    prefill_flops = 2 * num_params * sequence_length
    decode_flops = 2 * num_params * batch_size

    memory_bytes = num_params * bytes_per_param

    prefill_arithmetic_intensity = prefill_flops / memory_bytes
    decode_arithmetic_intensity = decode_flops / memory_bytes

    intensity_ridge = peak_flops / peak_bandwidth

    if prefill_arithmetic_intensity >= intensity_ridge:
        prefill_bottleneck = 'compute-bound'
        prefill_achieved_flops = peak_flops
    else:
        prefill_bottleneck = 'memory-bound'
        prefill_achieved_flops = prefill_arithmetic_intensity * peak_bandwidth

    if decode_arithmetic_intensity >= intensity_ridge:
        decode_bottleneck = 'compute-bound'
        decode_achieved_flops = peak_flops
    else:
        decode_bottleneck = 'memory-bound'
        decode_achieved_flops = decode_arithmetic_intensity * peak_bandwidth

    prefill_utilization_percent = float(np.round(100 * (prefill_achieved_flops / peak_flops), 2))
    decode_utilization_percent = float(np.round(100 * (decode_achieved_flops / peak_flops), 2))
    
    prefill = {
        'total_flops': prefill_flops,
        'memory_bytes': memory_bytes,
        'arithmetic_intensity': float(np.round(prefill_arithmetic_intensity, 1)),
        'bottleneck': prefill_bottleneck,
        'achieved_flops': float(np.round(prefill_achieved_flops, 1)),
        'utilization_percent': prefill_utilization_percent,
    }
    decode = {
        'total_flops': decode_flops,
        'memory_bytes': memory_bytes,
        'arithmetic_intensity': float(np.round(decode_arithmetic_intensity, 1)),
        'bottleneck': decode_bottleneck,
        'achieved_flops': float(np.round(decode_achieved_flops, 1)),
        'utilization_percent': decode_utilization_percent,
    }
    ridge_out = {
        "ridge_point": intensity_ridge,
        "prefill": prefill,
        "decode": decode,
    }

    return ridge_out