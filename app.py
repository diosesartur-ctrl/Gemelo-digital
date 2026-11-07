import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kalman import filtro_kalman
from modelos import comparar_modelos
from simulador import simular

st.set_page_config(page_title="Gemelo Digital GLP", page_icon="🚨", layout="wide")
st.title("Gemelo Digital: Detección de GLP")
st.caption("Simulación de sensores MQ-6 · Filtro de Kalman · Random Forest · Gradient Boosting · Alarma virtual")

pagina = st.sidebar.radio("Navegación", [
    "Simulación", "Señales", "Modelos predictivos", "Gemelo digital", "Resultados"
])

# ---------------------------------------------------------------- Parámetros
st.sidebar.header("Parámetros")
n = st.sidebar.slider("Duración (s)", 300, 1800, 900, 60)
inicio = st.sidebar.slider("Inicio de la 1.ª fuga (s)", 10, 200, 60)
episodios = st.sidebar.slider("Número de episodios de fuga", 1, 6, 4)
duracion_fuga = st.sidebar.slider("Duración de cada fuga (s)", 20, 300, 120, 10)
intensidad = st.sidebar.slider("Intensidad de fuga (ppm/s, ficticia)", 0.5, 8.0, 4.0, 0.1)
disipacion = st.sidebar.slider("Disipación por segundo", 0.001, 0.10, 0.015, 0.001, format="%.3f")
ruido = st.sidebar.slider("Ruido del sensor (ppm)", 0.0, 30.0, 9.0, 0.5)
distancia = st.sidebar.selectbox("Distancia del sensor (cm)", [40, 50, 60], index=1)
altura = st.sidebar.selectbox("Altura del sensor (cm)", [22, 42, 62], index=1)
angulo = st.sidebar.selectbox("Ángulo (°)", [-30, 0, 45], index=1)
horizonte = st.sidebar.slider("Horizonte de predicción (s)", 1, 20, 10)
umbral = st.sidebar.number_input("Umbral demostrativo (ppm)", min_value=1.0, value=147.0, step=1.0)

with st.sidebar.expander("Filtro de Kalman"):
    Q = st.number_input("Q (varianza de proceso)", min_value=0.001, value=1.5, step=0.5)
    R = st.number_input("R (varianza de medición)", min_value=0.001, value=81.0, step=5.0)
semilla = int(st.sidebar.number_input("Semilla aleatoria", value=42, step=1))

st.sidebar.caption("⚠️ 147 ppm es solo una referencia académica configurable, "
                   "no un umbral operativo certificado.")


# ---------------------------------------------------------------- Cálculo
@st.cache_data(show_spinner=False)
def generar(n, inicio, duracion_fuga, episodios, intensidad, disipacion, ruido, distancia, altura, angulo, semilla, Q, R):
    df = simular(n, inicio, duracion_fuga, episodios, intensidad, disipacion, ruido, distancia, altura, angulo, semilla)
    df["Kalman"] = filtro_kalman(df["Medida"].to_numpy(), Q, R)
    return df


@st.cache_data(show_spinner="Entrenando modelos...")
def entrenar(df, horizonte):
    return comparar_modelos(df, horizonte)


df = generar(n, inicio, duracion_fuga, episodios, intensidad, disipacion, ruido, distancia, altura, angulo, semilla, Q, R)


def linea_umbral(fig):
    fig.add_hline(y=umbral, line_dash="dash", line_color="red", annotation_text="Umbral demostrativo")


# ---------------------------------------------------------------- Páginas
if pagina == "Simulación":
    st.subheader("Señal del sensor MQ-6 (sintética)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Tiempo"], y=df["Medida"], name="Señal con ruido", opacity=0.6))
    fig.add_trace(go.Scatter(x=df["Tiempo"], y=df["Real_simulada"], name="Concentración simulada"))
    activa = df["Fuga_activa"].to_numpy().astype(int)
    cambios = np.flatnonzero(np.diff(np.r_[0, activa, 0]))
    for a, b in zip(cambios[::2], cambios[1::2]):
        fig.add_vrect(x0=a, x1=b, fillcolor="orange", opacity=0.12, line_width=0)
    linea_umbral(fig)
    fig.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Concentración (ppm)", height=450)
    st.caption("Franjas naranjas: episodios de fuga activa.")
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pico simulado", f"{df['Real_simulada'].max():.1f} ppm")
    c2.metric("Pico medido", f"{df['Medida'].max():.1f} ppm")
    c3.metric("Segundos sobre umbral (real)", int((df["Real_simulada"] >= umbral).sum()))

elif pagina == "Señales":
    st.subheader("Comparación de señales: medida vs Kalman")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Tiempo"], y=df["Medida"], name="Medida", opacity=0.5))
    fig.add_trace(go.Scatter(x=df["Tiempo"], y=df["Kalman"], name="Kalman"))
    fig.add_trace(go.Scatter(x=df["Tiempo"], y=df["Real_simulada"], name="Real simulada",
                             line=dict(dash="dot")))
    linea_umbral(fig)
    fig.update_layout(xaxis_title="Tiempo (s)", yaxis_title="ppm", height=450)
    st.plotly_chart(fig, width="stretch")

    err_med = df["Medida"] - df["Real_simulada"]
    err_kal = df["Kalman"] - df["Real_simulada"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE medida vs real", f"{np.sqrt((err_med ** 2).mean()):.2f} ppm")
    c2.metric("RMSE Kalman vs real", f"{np.sqrt((err_kal ** 2).mean()):.2f} ppm")
    reduccion = 1 - np.sqrt((err_kal ** 2).mean()) / max(np.sqrt((err_med ** 2).mean()), 1e-9)
    c3.metric("Reducción de error", f"{reduccion * 100:.1f} %")

    # Retraso: primer cruce del umbral (real vs Kalman)
    def primer_cruce(s):
        idx = np.where(s.to_numpy() >= umbral)[0]
        return int(df["Tiempo"].iloc[idx[0]]) if len(idx) else None
    t_real, t_kal = primer_cruce(df["Real_simulada"]), primer_cruce(df["Kalman"])
    if t_real is not None and t_kal is not None:
        st.info(f"Cruce del umbral — real: {t_real} s · Kalman: {t_kal} s · retraso: {t_kal - t_real} s")
    else:
        st.info("La señal no cruza el umbral con estos parámetros (sube la intensidad o baja el umbral).")

elif pagina == "Modelos predictivos":
    st.subheader("Comparación de algoritmos (20 % final como prueba)")
    resultados, preds = entrenar(df, horizonte)
    st.dataframe(resultados.style.format({"RMSE (ppm)": "{:.2f}", "MAE (ppm)": "{:.2f}", "R²": "{:.3f}"}),
                 hide_index=True, width="stretch")
    mejor = resultados.sort_values("RMSE (ppm)").iloc[0]
    st.success(f"Menor RMSE: **{mejor['Modelo']} / {mejor['Filtro']}** ({mejor['RMSE (ppm)']:.2f} ppm)")

    opcion = st.selectbox("Ver predicción de", list(preds))
    p = preds[opcion]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["Tiempo"], y=p["Real_futuro"], name=f"Real en t+{horizonte}s"))
    fig.add_trace(go.Scatter(x=p["Tiempo"], y=p["Prediccion"], name="Predicción"))
    linea_umbral(fig)
    fig.update_layout(xaxis_title="Tiempo de emisión t (s)", yaxis_title="ppm", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Versión demostrativa: señal artificial, una sola trayectoria. No valida capacidad predictiva general.")

elif pagina == "Gemelo digital":
    st.subheader("Monitoreo de emergencia")
    resultados, preds = entrenar(df, horizonte)
    opcion = st.selectbox("Modelo que alimenta la alarma", list(preds),
                          index=list(preds).index(
                              resultados.sort_values("RMSE (ppm)").iloc[0][["Modelo", "Filtro"]]
                              .str.cat(sep=" | ")))
    p = preds[opcion]
    i = st.slider("Instante de reproducción (dentro del tramo de prueba)", 0, len(p) - 1, len(p) - 1)
    fila = p.iloc[i]
    prediccion_ppm = float(fila["Prediccion"])
    alerta = prediccion_ppm >= umbral

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Concentración predicha (t+{horizonte}s)", f"{prediccion_ppm:.1f} ppm")
    c2.metric("Umbral", f"{umbral:.0f} ppm")
    c3.metric("Tiempo simulado", f"{int(fila['Tiempo'])} s")

    if alerta:
        st.error("🚨 ALERTA DE GAS")
        st.markdown("<h1 style='color:red;text-align:center;'>🔴 LUZ ACTIVADA</h1>", unsafe_allow_html=True)
    else:
        st.success("Condición normal")
        st.markdown("<h1 style='color:green;text-align:center;'>🟢 LUZ APAGADA</h1>", unsafe_allow_html=True)

    estado = {"simulacion": True, "tiempo_s": int(fila["Tiempo"]), "horizonte_s": horizonte,
              "modelo": opcion, "prediccion_ppm": round(prediccion_ppm, 2),
              "umbral_ppm": umbral, "luz_emergencia_virtual": bool(alerta)}
    with st.expander("Estado JSON"):
        st.json(estado)

    with st.expander("Publicar estado por MQTT (opcional)"):
        host = st.text_input("Host MQTT (TLS)", value="")
        topic = st.text_input("Topic", value="demo/glp/estado")
        port = st.number_input("Puerto", value=8883, min_value=1, max_value=65535)
        if st.button("Publicar estado virtual"):
            if not host:
                st.warning("Indica el host del broker.")
            else:
                try:
                    import paho.mqtt.publish as publish
                    cred = st.secrets.get("mqtt", {}) if "mqtt" in st.secrets else {}
                    auth = ({"username": cred["username"], "password": cred["password"]}
                            if "username" in cred and "password" in cred else None)
                    publish.single(topic, payload=json.dumps(estado), hostname=host, port=int(port),
                                   auth=auth, tls={}, retain=False)
                    st.success("Estado publicado (no es una orden de activación física).")
                except Exception as e:
                    st.error(f"No se pudo publicar: {e}")

    st.warning("La luz es exclusivamente virtual. Un sistema real debe usar detectores y alarmas "
               "certificados, independientes de la nube y de la predicción de ML.")

elif pagina == "Resultados":
    st.subheader("Métricas y exportación")
    resultados, preds = entrenar(df, horizonte)
    fig = go.Figure()
    for filtro in ["Sin Kalman", "Con Kalman"]:
        r = resultados[resultados["Filtro"] == filtro]
        fig.add_trace(go.Bar(x=r["Modelo"], y=r["RMSE (ppm)"], name=filtro))
    fig.update_layout(barmode="group", yaxis_title="RMSE (ppm)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(resultados.style.format({"RMSE (ppm)": "{:.2f}", "MAE (ppm)": "{:.2f}", "R²": "{:.3f}"}),
                 hide_index=True, width="stretch")

    todas = pd.concat([v.assign(Modelo=k) for k, v in preds.items()], ignore_index=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Señales CSV", df.to_csv(index=False).encode("utf-8"), "senales_glp.csv", "text/csv")
    c2.download_button("Métricas CSV", resultados.to_csv(index=False).encode("utf-8"), "metricas.csv", "text/csv")
    c3.download_button("Predicciones CSV", todas.to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")

st.divider()
st.caption("Simulación educativa: los datos son sintéticos, no mediciones reales ni resultados de validación.")
