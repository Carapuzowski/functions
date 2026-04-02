import numpy as np
from numpy.typing import NDArray

class MixedPrecision:
    def __init__(self, loss_scale: float=1024.0):
        self.loss_scale = loss_scale
    
    
    def forward(
            self,
            weights: NDArray,
            inputs: NDArray,
            targets: NDArray
        ) -> float:
        weights_fp16 = weights.astype(np.float16)
        inputs_fp16 = inputs.astype(np.float16)
        targets_fp16 = targets.astype(np.float16)

        predictions: NDArray[np.float16] = inputs_fp16 @ weights_fp16

        mse: NDArray[np.float32] = ((targets_fp16 - predictions)**2).mean().astype(np.float32)
        return float(mse * self.loss_scale)

    
    def backward(self, gradients: NDArray):
        grandients_fp32 = gradients.astype(np.float32)
        unscaled_gradients = grandients_fp32 / self.loss_scale
        
        overflow = np.isnan(unscaled_gradients).any() or np.isinf(unscaled_gradients).any()
        s
        if overflow:
            unscaled_gradients = np.zeros_like(unscaled_gradients)
            
        return unscaled_gradients