import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ==============================================================================
# 1. Dataset de Entrenamiento Ampliado: [Edad, Salario (miles), Número de Hijos]
# (Misión Práctica - Puntos 2 y 3: al menos 10 filas y 3 dimensiones)
# ==============================================================================
X_entrenamiento = np.array([
    [20, 30, 0],  # Cliente 1
    [40, 50, 2],  # Cliente 2
    [35, 45, 1],  # Cliente 3
    [22, 25, 0],  # Cliente 4
    [45, 60, 3],  # Cliente 5
    [28, 38, 1],  # Cliente 6
    [50, 70, 2],  # Cliente 7
    [23, 32, 0],  # Cliente 8
    [38, 48, 2],  # Cliente 9
    [60, 80, 1]   # Cliente 10
])

# Etiquetas de clase: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

# Instancia a clasificar: [Edad: 30, Salario: 40, Hijos: 1]
nuevo_cliente = np.array([[30, 40, 1]])

# ==============================================================================
# 2. Experimentación con K = 1 (Misión Práctica - Punto 4)
# ==============================================================================
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = modelo_k1.predict(nuevo_cliente)
resultado_k1 = "COMPRA" if pred_k1[0] == 1 else "NO COMPRA"
print(f"[Experimento K=1] Clase predicha: {pred_k1[0]} -> {resultado_k1}")

# ==============================================================================
# 3. Experimentación con K = 5 (Misión Práctica - Punto 4)
# ==============================================================================
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = modelo_k5.predict(nuevo_cliente)
resultado_k5 = "COMPRA" if pred_k5[0] == 1 else "NO COMPRA"
print(f"[Experimento K=5] Clase predicha: {pred_k5[0]} -> {resultado_k5}")