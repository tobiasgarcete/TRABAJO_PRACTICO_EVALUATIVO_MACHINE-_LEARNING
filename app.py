from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from predict import predecir_nivel

app = FastAPI()

# Permitir CORS para que el frontend pueda hacer peticiones (ajusta el origen si lo necesitas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o coloca tu dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DatosEntrada(BaseModel):
    sueno_horas: float
    cafe_tazas: int
    ejercicio_min: int
    pantallas_horas: float
    estudio_horas: float
    comida_chatarra: int

@app.post("/predecir_estres/")
def predecir_estres(datos: DatosEntrada):
    entrada = datos.dict()
    resultado = predecir_nivel(entrada)

    # Puedes ajustar las descripciones para enviar en la respuesta
    niveles_desc = {0: "Bajo", 1: "Medio", 2: "Alto"}

    nivel_esperado = niveles_desc[resultado["nivel_estres"].count("🟢") > 0 and 0 or (resultado["nivel_estres"].count("⚠️") > 0 and 1 or 2)]
    nivel_predicho = niveles_desc.get(resultado["nivel_modelo"], "Desconocido")

    coincidencia = (resultado["nivel_estres"][0] == nivel_predicho[0])

    return {
        "score_calculado": resultado["score"],
        "nivel_esperado": {"codigo": resultado["nivel_estres"][0], "descripcion": nivel_esperado},
        "nivel_predicho": {"codigo": resultado["nivel_modelo"], "descripcion": nivel_predicho},
        "coincidencia": coincidencia,
        "precision": 0.85  # Puedes pasar la precisión real si la guardaste
    }