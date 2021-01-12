import numpy as np

def rmspcte(real, pred):
    """Implements Root Mean Squared Percent Error (excluding zero sales)"""
    mask = ~(real == 0)  # exclude zero sales (see eval rules)
    real, pred = real[mask], pred[mask]
    return np.sqrt(np.mean(((real.values-pred.values)/real.values)**2))**0.5