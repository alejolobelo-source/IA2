import numpy as np
from sklearn.svm import SVC

# ==============================================================================
# PARTE 1: Dataset Original y Verificación de Vectores de Soporte
# ==============================================================================
X = np.array([[2,2], [3,3], [4,2], [6,6], [7,8], [8,7]])
Y = np.array([0, 0, 0, 1, 1, 1])

# Modelo lineal inicial
modelo_svm = SVC(kernel='linear')
modelo_svm.fit(X, Y)

print("--- EXPERIMENTO 1: DATASET ORIGINAL (LINEAL) ---")
print("Vectores de Soporte detectados por Scikit-Learn:\n", modelo_svm.support_vectors_)

nuevo_punto = np.array([[5, 4]])
pred = modelo_svm.predict(nuevo_punto)
print(f"Prediccion para el punto [5, 4]: Clase {pred[0]}")

# ==============================================================================
# PARTE 2: Agregar Punto [5,5] con Clase 0 (Misión Práctica 2, 3 y 4)
# ==============================================================================
X_mod = np.vstack([X, [5, 5]])
Y_mod = np.append(Y, 0)  # Agregamos etiqueta 0

print("\n--- EXPERIMENTO 2: PUNTO ANÓMALO [5,5] CON KERNEL LINEAL ---")
svm_lineal_mod = SVC(kernel='linear')
svm_lineal_mod.fit(X_mod, Y_mod)
print("Vectores de soporte forzados (Lineal):\n", svm_lineal_mod.support_vectors_)
print("Prediccion [5, 5] con Lineal:", svm_lineal_mod.predict([[5, 5]])[0])

print("\n--- EXPERIMENTO 3: KERNEL RBF (FUNCION DE BASE RADIAL) ---")
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_mod, Y_mod)
print("Vectores de soporte con RBF:\n", svm_rbf.support_vectors_)
print("Prediccion [5, 5] con RBF:", svm_rbf.predict([[5, 5]])[0])