import numpy as np

# Función de Activación: Sigmoide (devuelve un valor entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# ==============================================================================
# 1. CÓDIGO BASE TAL CUAL LA GUÍA (Página 4) - 1 Cliente
# ==============================================================================
# ENTRADA (X): 1 cliente con 3 características
X = np.array([0.5, 0.8, 0.2])

# CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas)
W1 = np.array([
    [ 0.1,  0.2,  0.3,  0.4],
    [-0.5,  0.6,  0.7, -0.8],
    [ 0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 Sesgos

# PROCESO CAPA OCULTA
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)  # Salida de la capa oculta

# CAPA DE SALIDA (1 Neurona)
# Matriz W2 de (4 entradas ocultas x 1 neurona final)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# PROCESO CAPA FINAL (con el signo + corregido)
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("--- MISIÓN PRÁCTICA: PASOS 1 Y 2 ---")
print("Variable Z1 (valores lineales crudos):", np.round(Z1, 4))
print("Variable A1 (activación en rango 0 a 1):", np.round(A1, 4))
print("Predicción de la Red (Probabilidad):", np.round(Salida_Final[0], 4))

# ==============================================================================
# 2. EL RETO DIMENSIONAL (Página 5) - Procesamiento en Lote / Batch
# ==============================================================================
print("\n--- MISIÓN PRÁCTICA: PASOS 3 Y 4 (RETO DIMENSIONAL) ---")
# Matriz de 2x3 para procesar 2 clientes simultáneamente
X_batch = np.array([
    [0.5, 0.8, 0.2],
    [0.1, 0.9, 0.9]
])

# Propagación matricial simultánea (sin modificar W1, b1, W2 ni b2)
Z1_batch = np.dot(X_batch, W1) + b1
A1_batch = sigmoide(Z1_batch)
Z2_batch = np.dot(A1_batch, W2) + b2
Salidas_Batch = sigmoide(Z2_batch)

print("Predicción Cliente 1 (Probabilidad):", np.round(Salidas_Batch[0], 4))
print("Predicción Cliente 2 (Probabilidad):", np.round(Salidas_Batch[1], 4))