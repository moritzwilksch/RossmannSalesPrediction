from tensorflow.keras import backend as K
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


def rmspe_loss(y_true, y_pred):
    """RMSPE loss on keras backend."""
    sum = K.sqrt(K.mean(K.square((y_true - y_pred) /
                                 K.clip(K.abs(y_true), K.epsilon(), None) + 1e-6), axis=-1))
    return sum*100
