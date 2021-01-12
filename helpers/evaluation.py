import numpy as np

def rmspcte(real, pred):
    """Implements Root Mean Squared Percent Error (excluding zero sales)"""
    mask = ~(real == 0)  # exclude zero sales (see eval rules)
    real, pred = real[mask], pred[mask]

    if type(real) != np.ndarray:
        real = real.values

    if type(pred) != np.ndarray:
        pred = pred.values

    real, pred = real.flatten(), pred.flatten()

    return np.sqrt(np.mean(((real-pred)/real)**2))