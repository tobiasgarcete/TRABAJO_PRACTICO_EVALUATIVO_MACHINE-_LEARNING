# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from utils import calcular_score, calcular_estres

np.random.seed(42)
n = 600

# Generar datos simulados
data = pd.DataFrame({
    "sueno_horas": np.random.uniform(3, 9, n),
    "cafe_tazas": np.random.randint(0, 5, n),
    "ejercicio_min": np.random.randint(0, 120, n),
    "pantallas_horas": np.random.uniform(1, 12, n),
    "estudio_horas": np.random.uniform(0, 10, n),
    "comida_chatarra": np.random.randint(0, 4, n),
})

# Calcular score y estrés
data['score'] = data.apply(calcular_score, axis=1)
data['estres'] = data.apply(lambda row: calcular_estres(row['score'], data), axis=1)

# Balancear clases
min_count = data['estres'].value_counts().min()
data_balanceada = data.groupby('estres', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))

X = data_balanceada.drop(columns=['estres', 'score'])
y = data_balanceada['estres']

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar precisión
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {acc:.2f}")

# Guardar modelo y metadatos
modelo_info = {
    "model": model,
    "p25": data['score'].quantile(0.25),
    "p75": data['score'].quantile(0.75),
    "accuracy": acc
}
joblib.dump(modelo_info, "modelo_estres.pkl")
print("Modelo guardado en 'modelo_estres.pkl'")
