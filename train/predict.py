import joblib
import pandas as pd

# Cargar modelo + cortes reales
modelo_info = joblib.load("modelo_estres.pkl")
model = modelo_info["model"]
p25 = modelo_info["p25"]
p75 = modelo_info["p75"]

def calcular_score(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.5
    score += row["cafe_tazas"] * 0.4
    score -= row["ejercicio_min"] * 0.005
    score += row["pantallas_horas"] * 0.2
    score -= row["estudio_horas"] * 0.01
    score += row["comida_chatarra"] * 0.8
    return score

def predecir_nivel(datos_entrada):
    df_input = pd.DataFrame([datos_entrada])
    score_input = calcular_score(df_input.iloc[0])

    if score_input < p25:
        nivel_esperado = "Bajo 🟢"
        codigo_esperado = 0
    elif score_input < p75:
        nivel_esperado = "Medio ⚠️"
        codigo_esperado = 1
    else:
        nivel_esperado = "Alto 🔴"
        codigo_esperado = 2

    pred = model.predict(df_input)[0]
    niveles = {0: "Bajo 🟢", 1: "Medio ⚠️", 2: "Alto 🔴"}

    print("\nNiveles de Estrés:")
    print("0: Bajo 🟢 (sin signos de estrés o muy leves)")
    print("1: Medio ⚠️ (estrés moderado o puntual)")
    print("2: Alto 🔴 (estrés elevado que puede afectar el bienestar)")

    print(f"\nScore calculado manualmente: {score_input:.2f}")
    print(f"Nivel esperado según score manual: {nivel_esperado} (Código {codigo_esperado})")
    print(f"Nivel predicho por el modelo: {niveles[pred]} (Código {pred})")

    if codigo_esperado == pred:
        print("✅ El modelo predijo correctamente el nivel esperado.")
    else:
        print("❌ El modelo NO coincidió con el nivel esperado.")

if __name__ == "__main__":
    entrada = {
        "sueno_horas": 6,
        "cafe_tazas": 2,
        "ejercicio_min": 30,
        "pantallas_horas": 5,
        "estudio_horas": 2,
        "comida_chatarra": 2,
    }
    predecir_nivel(entrada)
