import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Semilla
seed = 42

# Cargar el dataset original
iris_df = pd.read_csv("data/iris.csv")
iris_df.sample(frac=1, random_state=seed)

# Seleccionar las características y las especies
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# Separar el conjunto de datos en training y tesT
# 70% entrenamiento and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# Definir el clasificadotes
clf = RandomForestClassifier(n_estimators=100)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predicción en el conjunto de test
y_pred = clf.predict(X_test)

# Calcular accurcy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# Guardamos en el modelo en el disco
joblib.dump(clf, "rf_model.sav")
