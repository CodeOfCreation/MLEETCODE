import numpy as np

def normalize_3d(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, 3)

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    normalized = np.divide(v, norms, where=norms > 1e-12)
    normalized[norms.squeeze() <= 1e-12] = 0.0
    return normalized if v.shape[0] > 1 else normalized[0]