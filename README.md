# Predicción de Nivel de Estrés en Estudiantes

Este proyecto es una aplicación web desarrollada con **FastAPI** que predice el nivel de estrés en estudiantes a partir de hábitos diarios como horas de sueño, actividad física, alimentación, entre otros.

## 🧠 Descripción

El objetivo es ayudar a identificar posibles niveles de estrés a través de un modelo de machine learning entrenado con un dataset relevante, y mostrar visualmente el resultado al usuario mediante una interfaz simple.

## 🚀 Tecnologías utilizadas

- [FastAPI](https://fastapi.tiangolo.com/) – Framework backend
- [Scikit-learn](https://scikit-learn.org/) – Entrenamiento del modelo
- [Uvicorn](https://www.uvicorn.org/) – Servidor ASGI para correr la API
- HTML + JS (Frontend simple)
- Pandas y NumPy – Procesamiento de datos


## ⚙️ Instalación y ejecución

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd trabajo_final_grupal_python
```
## Crea y activa un entorno virtual:

bash
python -m venv .venv
.venv\Scripts\activate    En Windows

## Instala las dependencias:

bash
pip install -r requirements.txt

## Ejecuta el servidor:

bash
python -m uvicorn app:app --reload
