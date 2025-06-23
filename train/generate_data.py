# train/generate_data.py
import numpy as np
import pandas as pd
from utils.features import calcular_score, calcular_estres

np.random.seed(42)
n = 600

# Generar datos sintéticos
data = pd.DataFrame({
    "sueno_horas": np.random.uniform(3, 9, n),
    "cafe_tazas": np.random.randint(0, 5, n),
    "ejercicio_min": np.random.randint(0, 120, n),
    "pantallas_horas": np.random.uniform(1, 12, n),
    "estudio_horas": np.random.uniform(0, 10, n),
    "comida_chatarra": np.random.randint(0, 4, n),
})

# Calcular score y niveles
data["score"] = data.apply(calcular_score, axis=1)
p25 = data["score"].quantile(0.25)
p75 = data["score"].quantile(0.75)
data["estres"] = data["score"].apply(lambda x: calcular_estres(x, p25, p75))

# Balancear las clases
min_count = data["estres"].value_counts().min()
data_balanceada = data.groupby("estres").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Guardar si querés inspeccionar después
data_balanceada.to_csv("data/synthetic_data.csv", index=False)

print(data_balanceada["estres"].value_counts())
