# Predicción de Estrés - Proyecto ML con FastAPI y Frontend

Este proyecto es una aplicación web para predecir el nivel de estrés en estudiantes a partir de sus hábitos diarios. Usa un modelo de Machine Learning servido con FastAPI y un frontend simple para mostrar resultados con colores, emojis y un gráfico visual.

---

## Estructura

- `estres_api.py`: Archivo backend con FastAPI que expone el endpoint `/predecir_estres/` para recibir datos y devolver la predicción.
- `index.html`: Página web con formulario para ingresar hábitos, que consume la API y muestra resultados.

---

## Requisitos previos

- Python 3.8+
- Node.js o servidor web estático para servir el frontend (puede ser live-server o simplemente abrir `index.html` en navegador)
- `uvicorn` para correr FastAPI

---

## Instalación y ejecución

1. Clona este repositorio o descarga los archivos.

2. Instala las dependencias de Python:

```bash
pip install fastapi uvicorn fastapi scikit-learn
```
## Ejecuta el servidor FastAPI:

python -m uvicorn estres_api:app --reload


## 👤 Autor
Trabajo práctico evaluativo - Machine Learning
Alumno: [Tobias Garcete Lionel
         Acosta Mirko Josue Ian]
Profesor/a: [Gabriel Acosta
             Agustin Mazza]
