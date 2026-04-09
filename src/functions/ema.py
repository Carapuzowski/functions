import numpy as np

def ema_update(ema_params: np.ndarray, model_params_list: list[np.ndarray], decay: float) -> np.ndarray:
    """
    Compute the Exponential Moving Average of model parameters over training steps.
    
    Args:
        ema_params: numpy array, initial EMA parameters
        model_params_list: list of numpy arrays, model params at each training step
        decay: float, EMA decay rate in [0, 1]
    Returns:
        Final EMA parameters as a (nested) list, rounded to 4 decimal places
    """
    temp_memory: np.ndarray = ema_params
    for params in model_params_list:
        temp_memory *= decay
        temp_memory += (1-decay)*params
    return temp_memory.round(4).tolist()

if __name__=="__main__":
    ema_params = np.array([1.0, 2.0, 3.0])
    model_params_list = [np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])]
    decay = 0.9 
    ema_update(ema_params, model_params_list, decay)