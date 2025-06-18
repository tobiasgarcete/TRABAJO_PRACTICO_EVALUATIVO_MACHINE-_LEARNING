import numpy as np
import pandas as pd

np.random.seed(42)
n = 600

data = pd.DataFrame({
    "sueno_horas": np.random.uniform(3, 9, n),
    "cafe_tazas": np.random.randint(0, 5, n),
    "ejercicio_min": np.random.randint(0, 120, n),
    "pantallas_horas": np.random.uniform(1, 12, n),
    "estudio_horas": np.random.uniform(0, 10, n),
    "comida_chatarra": np.random.randint(0, 4, n),
})

def calcular_estres(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.3
    score += row["cafe_tazas"] * 0.2
    score -= row["ejercicio_min"] * 0.01
    score += row["pantallas_horas"] * 0.1
    score -= row["estudio_horas"] * 0.05
    score += row["comida_chatarra"] * 0.4

    if score < 2.5:
        return 0  
    elif score < 5:
        return 1  
    else:
        return 2  

data["estres"] = data.apply(calcular_estres, axis=1)

min_count = data['estres'].value_counts().min()
data_balanceada = data.groupby("estres").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

print(data_balanceada['estres'].value_counts())
