import numpy as np

# 1. Definir la Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Producto punto (Combinación lineal)
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida

# ==========================================================
# RETO RESUELTO: COMPUERTA LÓGICA OR
# ==========================================================
pesos = np.array([0.5, 0.5])  # Vector W
sesgo = -0.2                  # Constante b calibrada para OR

# Comprobación de todas las entradas del OR
casos = [
    np.array([1, 1]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, 0])
]

print("--- RESULTADOS COMPUERTA OR ---")
for entrada in casos:
    resultado = perceptron(entrada, pesos, sesgo)
    print(f"Entrada: {entrada} -> El Perceptrón disparó el valor: {resultado}")