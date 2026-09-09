import numpy as np

# =======================================================
# TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES
# =======================================================
print("=== TALLER DE LABORATORIO 1 ===")

# 1. Matriz 5x5 sobreexpuesta [200, 255)
A = np.random.randint(200, 255, (5, 5))
print("Matriz Original (Sobreexpuesta):\n", A)

# 2. Reducción de contraste del 50% (alpha = 0.5) y brillo en 50 (beta = -50.0)
alpha = 0.5
beta = -50.0
A_nueva = alpha * A + beta

# 3. Acotamiento (Clipping) y conversión a np.uint8
A_nueva = np.clip(A_nueva, 0, 255).astype(np.uint8)

# 4. Resultado final
print("\nMatriz Procesada:\n", A_nueva)


# =======================================================
# TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL
# =======================================================
print("\n=== TALLER DE LABORATORIO FINAL ===")

# 1. Declaración de Sección de Imagen (I) y Kernel (K)
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
])

K = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# 2. Producto Hadamard y suma de todos los valores
producto_hadamard = I * K
pixel_central = np.sum(producto_hadamard)

# 3. Impresión del valor resultando para el píxel central (Resultado esperado: 600)
print("Matriz Producto Hadamard:\n", producto_hadamard)
print("\nValor calculado del píxel central:", pixel_central)