# predict.py
import joblib
import pandas as pd
from utils.features import calcular_score, calcular_estres

# Cargar modelo y percentiles
modelo_info = joblib.load("model/modelo_estres.pkl")
model = modelo_info["model"]
p25 = modelo_info["p25"]
p75 = modelo_info["p75"]

def predecir_nivel(datos_entrada):
    """
    datos_entrada: dict con claves:
      - sueno_horas, cafe_tazas, ejercicio_min, pantallas_horas, estudio_horas, comida_chatarra
    """
    df_input = pd.DataFrame([datos_entrada])

    score_input = calcular_score(df_input.iloc[0])
    nivel_codificado = calcular_estres(score_input, p25, p75)

    niveles = {0: "Bajo 🟢", 1: "Medio ⚠️", 2: "Alto 🔴"}
    nivel_str = niveles.get(nivel_codificado, "Desconocido")

    pred_modelo = model.predict(df_input)[0]

    return {
        "score": round(score_input, 2),
        "nivel_estres": nivel_str,
        "nivel_modelo": int(pred_modelo)
    }
