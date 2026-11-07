"""Comparación Random Forest vs Gradient Boosting con y sin filtro de Kalman.

Partición temporal: 80 % inicial entrenamiento, 20 % final prueba (sin barajar).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def crear_variables(df, senal, horizonte):
    s = df[senal].reset_index(drop=True)
    X = pd.DataFrame({
        "nivel": s,
        "lag1": s.shift(1), "lag2": s.shift(2), "lag3": s.shift(3), "lag5": s.shift(5),
        "media5": s.rolling(5).mean(), "media12": s.rolling(12).mean(),
        "pendiente3": s - s.shift(3), "pendiente8": s - s.shift(8),
        "desv5": s.rolling(5).std(),
    })
    X["objetivo"] = df["Real_simulada"].reset_index(drop=True).shift(-horizonte)
    X["Tiempo"] = df["Tiempo"].reset_index(drop=True)
    return X.dropna().reset_index(drop=True)


def _modelos():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                                           l2_regularization=1.0, random_state=42),
    }


def comparar_modelos(df, horizonte=10, proporcion_train=0.8):
    resultados, predicciones = [], {}
    for senal, filtro in [("Medida", "Sin Kalman"), ("Kalman", "Con Kalman")]:
        datos = crear_variables(df, senal, horizonte)
        corte = int(len(datos) * proporcion_train)
        train, test = datos.iloc[:corte], datos.iloc[corte:]
        cols = [c for c in datos.columns if c not in ("objetivo", "Tiempo")]
        for nombre, modelo in _modelos().items():
            modelo.fit(train[cols], train["objetivo"])
            pred = np.maximum(0, modelo.predict(test[cols]))
            y = test["objetivo"].to_numpy()
            resultados.append({
                "Modelo": nombre, "Filtro": filtro,
                "RMSE (ppm)": float(np.sqrt(mean_squared_error(y, pred))),
                "MAE (ppm)": float(mean_absolute_error(y, pred)),
                "R²": float(r2_score(y, pred)),
            })
            predicciones[f"{nombre} | {filtro}"] = pd.DataFrame(
                {"Tiempo": test["Tiempo"].to_numpy(), "Real_futuro": y, "Prediccion": pred})
    return pd.DataFrame(resultados), predicciones
