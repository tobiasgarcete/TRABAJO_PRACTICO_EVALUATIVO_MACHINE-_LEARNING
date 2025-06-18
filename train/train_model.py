import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

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
# Calcular percentiles 25 y 75 para cada columna numérica
percentiles = data.quantile([0.25, 0.75])
print("Percentiles 25 y 75 de las variables:")
print(percentiles)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)
n = 600

# Generación de datos
data = pd.DataFrame({
    "sueno_horas": np.random.uniform(3, 9, n),
    "cafe_tazas": np.random.randint(0, 5, n),
    "ejercicio_min": np.random.randint(0, 120, n),
    "pantallas_horas": np.random.uniform(1, 12, n),
    "estudio_horas": np.random.uniform(0, 10, n),
    "comida_chatarra": np.random.randint(0, 4, n),
})

# Función de score
def calcular_score(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.5
    score += row["cafe_tazas"] * 0.4
    score -= row["ejercicio_min"] * 0.005
    score += row["pantallas_horas"] * 0.2
    score -= row["estudio_horas"] * 0.01
    score += row["comida_chatarra"] * 0.8
    return score

data['score'] = data.apply(calcular_score, axis=1)

# Percentiles de corte
p25 = data['score'].quantile(0.25)
p75 = data['score'].quantile(0.75)

def calcular_estres(row):
    score = row['score']
    if score < p25:
        return 0  # Bajo
    elif score < p75:
        return 1  # Medio
    else:
        return 2  # Alto

data["estres"] = data.apply(calcular_estres, axis=1)

# Balanceo de clases
min_count = data['estres'].value_counts().min()
data_balanceada = data.groupby("estres", group_keys=False).apply(lambda x: x.sample(min_count))

X = data_balanceada.drop(columns=["estres", "score"])
y = data_balanceada["estres"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nAccuracy: {acc:.2f}")

# Guardar modelo + percentiles juntos
modelo_info = {
    "model": model,
    "p25": p25,
    "p75": p75
}
joblib.dump(modelo_info, "modelo_estres.pkl")
print(f"Modelo y percentiles guardados correctamente.")

# Por ejemplo, podés hacer un chequeo simple:
print("\nEjemplo de análisis simple con percentiles:")
for col in data.columns:
    p25 = percentiles.loc[0.25, col]
    p75 = percentiles.loc[0.75, col]
    print(f"{col}: P25={p25:.2f}, P75={p75:.2f}")

    # Por ejemplo, contar cuántos registros están por encima del P75 en esa variable
    count_high = (data[col] > p75).sum()
    print(f"  Registros con {col} > P75: {count_high}")

# Calcular score manual para todo el dataset
def calcular_score(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.5
    score += row["cafe_tazas"] * 0.4
    score -= row["ejercicio_min"] * 0.005
    score += row["pantallas_horas"] * 0.2
    score -= row["estudio_horas"] * 0.01
    score += row["comida_chatarra"] * 0.8
    return score

data['score'] = data.apply(calcular_score, axis=1)

print("Estadísticas del score:")
print(data['score'].describe())

# Definir nuevos cortes con base en percentiles
p25 = data['score'].quantile(0.25)
p75 = data['score'].quantile(0.75)

def calcular_estres(row):
    score = row['score']
    if score < p25:
        return 0  # Bajo
    elif score < p75:
        return 1  # Medio
    else:
        return 2  # Alto

data["estres"] = data.apply(calcular_estres, axis=1)

print("\nDistribución original de niveles de estrés:")
print(data['estres'].value_counts())

# Balancear las clases
min_count = data['estres'].value_counts().min()
data_balanceada = data.groupby("estres", group_keys=False).apply(lambda x: x.sample(min_count))

print("\nDistribución balanceada:")
print(data_balanceada['estres'].value_counts())

X_bal = data_balanceada.drop(columns=["estres", "score"])
y_bal = data_balanceada["estres"]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar precisión
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en test con datos balanceados: {acc:.2f}")

joblib.dump(model, "modelo_estres.pkl")
print("Modelo guardado en 'modelo_estres.pkl'")

# Función para predecir y mostrar resultados comparativos
def predecir_nivel(datos_entrada):
    df_input = pd.DataFrame([datos_entrada])
    score_input = calcular_score(df_input.iloc[0])

    # Nivel esperado con cortes
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

# Ejemplo de prueba
input_test = {
    "sueno_horas": 6,
    "cafe_tazas": 2,
    "ejercicio_min": 30,
    "pantallas_horas": 5,
    "estudio_horas": 2,
    "comida_chatarra": 2,
}

predecir_nivel(input_test)