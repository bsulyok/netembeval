import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dist = np.subtract(x, y)
    dist = np.power(dist, 2)
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    return dist


def native_disk_distance(x: np.ndarray, y: np.ndarray):
    x0, y0 = np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1)
    cosim = np.sum(np.multiply(x, y), axis=-1) / x0 / y0
    cosh_dist = np.cosh(x0) * np.cosh(y0) - np.sinh(x0) * np.sinh(y0) * cosim
    cosh_dist = cosh_dist.clip(min=1.0)
    dist = np.arccosh(cosh_dist)
    return dist