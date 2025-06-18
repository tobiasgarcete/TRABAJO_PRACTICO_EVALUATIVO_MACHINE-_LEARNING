from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Agregar CORS para permitir peticiones desde http://127.0.0.1:5500 (Live Server de VSCode)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",  # 👈 necesario para Live Server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # puedes usar ["*"] si es solo para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y percentiles al iniciar el servidor
modelo_info = joblib.load("modelo_estres.pkl")
model = modelo_info["model"]
p25 = modelo_info["p25"]
p75 = modelo_info["p75"]

# Esquema de entrada
class DatosEntrada(BaseModel):
    sueno_horas: float
    cafe_tazas: int
    ejercicio_min: int
    pantallas_horas: float
    estudio_horas: float
    comida_chatarra: int

# Función para calcular el score personalizado
def calcular_score(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.5
    score += row["cafe_tazas"] * 0.4
    score -= row["ejercicio_min"] * 0.005
    score += row["pantallas_horas"] * 0.2
    score -= row["estudio_horas"] * 0.01
    score += row["comida_chatarra"] * 0.8
    return score

# Endpoint principal
@app.post("/predecir_estres/")
def predecir_estres(datos: DatosEntrada):
    df_input = pd.DataFrame([datos.dict()])
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

    pred = int(model.predict(df_input)[0])  # 👈 Convertir a int para evitar error de serialización
    niveles = {0: "Bajo 🟢", 1: "Medio ⚠️", 2: "Alto 🔴"}

    return {
        "score_calculado": round(score_input, 2),
        "nivel_esperado": {"codigo": codigo_esperado, "descripcion": nivel_esperado},
        "nivel_predicho": {"codigo": pred, "descripcion": niveles.get(pred, "Desconocido")},
        "coincidencia": codigo_esperado == pred
    }
