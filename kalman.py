"""Filtro de Kalman escalar causal (solo usa datos pasados y presentes)."""
import numpy as np


def filtro_kalman(z, Q=1.5, R=81.0):
    z = np.asarray(z, dtype=float)
    x = float(z[0])
    P = max(R, 1.0)
    salida = np.empty_like(z)
    for i, medida in enumerate(z):
        P = P + Q              # predicción
        K = P / (P + R)        # ganancia
        x = x + K * (medida - x)
        P = (1 - K) * P        # actualización
        salida[i] = x
    return salida
