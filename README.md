# Gemelo Digital GLP 🚨

Interfaz web (Streamlit) que simula señales de un sensor MQ-6 de GLP, las filtra con Kalman, compara Random Forest y Gradient Boosting, y muestra una luz de emergencia **virtual** con umbral configurable.

## Estructura

```
gemelo_glp/
├── app.py              # Interfaz (5 secciones)
├── simulador.py        # Señales sintéticas con episodios de fuga
├── kalman.py           # Filtro de Kalman escalar causal
├── modelos.py          # RF vs GB, con y sin Kalman (80/20 temporal)
├── requirements.txt
├── .streamlit/config.toml
└── datos/
```

## Secciones

| Sección | Qué hace |
|---|---|
| Simulación | Configura fuga, ruido, posición del sensor y tiempo |
| Señales | Medida vs Kalman, reducción de error y retraso de detección |
| Modelos predictivos | RMSE, MAE y R² de RF y GB, con y sin Kalman |
| Gemelo digital | Predicción a t+h y luz virtual (opcional: publicar por MQTT) |
| Resultados | Gráfico comparativo y descarga de CSV |

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre http://localhost:8501

## MQTT (opcional)

En Streamlit Cloud: *Settings → Secrets*:

```toml
[mqtt]
username = "USUARIO"
password = "CLAVE"
```

Nunca subas credenciales al repositorio.

## Advertencias

- Los datos son **sintéticos**; no son mediciones reales ni resultados de validación.
- 147 ppm es una referencia académica configurable, no un umbral de seguridad certificado.
- La luz es virtual. Un sistema real debe usar detectores y alarmas certificados, independientes de la nube y del modelo de ML.
