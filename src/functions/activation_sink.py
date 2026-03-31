import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
	"""
	Detect attention sink tokens from multi-head attention weight matrices.
	
	Args:
		attn_weights: Attention weights of shape (num_heads, seq_len, seq_len)
		threshold: Minimum average received attention to qualify as a sink
		
	Returns:
		Dictionary with 'sink_positions', 'avg_attention_received', and 'sink_scores'
	"""
	avg_attention_received = np.round(np.mean(attn_weights, axis=(0,1)), 4)
	sink_positions_mask = avg_attention_received > threshold
	sink_positions = np.where(sink_positions_mask)[0]

	sink_positions = np.sort(sink_positions)

	sink_scores = avg_attention_received[sink_positions]

	return {
		'sink_positions': sink_positions.tolist(),
		'avg_attention_received': avg_attention_received.tolist(),
		'sink_scores': sink_scores.tolist(),
	}