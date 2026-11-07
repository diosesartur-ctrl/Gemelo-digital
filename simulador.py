"""Generación de señales sintéticas de concentración de GLP (sensor MQ-6 simulado).

Se generan varios episodios de fuga para que tanto el tramo de entrenamiento
como el de prueba contengan subidas de concentración.
Los factores geométricos son ilustrativos y NO están calibrados con datos reales.
"""
import numpy as np
import pandas as pd


def simular(n=900, inicio_fuga=60, duracion_fuga=120, episodios=4, intensidad=4.0,
            disipacion=0.015, ruido=9.0, distancia=50, altura=42, angulo=0, semilla=42):
    rng = np.random.default_rng(semilla)
    t = np.arange(n, dtype=int)
    # Atenuación por posición del sensor (solo didáctica)
    geo = (50 / distancia) ** 1.15 * (1.0 if altura == 42 else 0.72) * (1.0 if angulo == 0 else 0.83)

    # Episodios de fuga equiespaciados, cada uno con intensidad ligeramente distinta
    periodo = (n - inicio_fuga) / episodios
    fuga = np.zeros(n)
    for k in range(episodios):
        a = int(inicio_fuga + k * periodo)
        b = min(n, a + int(min(duracion_fuga, periodo - 10)))
        fuga[a:b] = intensidad * rng.uniform(0.75, 1.25)

    real = np.zeros(n)
    for i in range(1, n):
        real[i] = max(0.0, real[i - 1] * (1 - disipacion) + fuga[i] * geo)
    medida = np.maximum(0.0, real + rng.normal(0, ruido, n))
    return pd.DataFrame({"Tiempo": t, "Real_simulada": real, "Medida": medida, "Fuga_activa": fuga > 0})
