import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen RGB/BGR
# (Usa una imagen descargada o genera una sintética para pruebas)
filas, columnas = 400, 600
imagen = np.zeros((filas, columnas, 3), dtype=np.uint8)
imagen[:, :, 0] = np.random.randint(50, 120, (filas, columnas))  # Azul (B)
imagen[:, :, 1] = np.random.randint(
    150, 255, (filas, columnas)
)  # Verde (G) Predominante
imagen[:, :, 2] = np.random.randint(20, 80, (filas, columnas))  # Rojo (R)

# 2. Separar la imagen en sus 3 canales mediante Slicing
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 2]

# 3. Calcular el histograma de cada canal por separado
hist_b = cv2.calcHist([imagen], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([imagen], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([imagen], [2], None, [256], [0, 256])

# 4. Graficar los tres histogramas superpuestos con matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist_b, color='blue', label='Canal Azul (B)')
plt.plot(hist_g, color='green', label='Canal Verde (G)')
plt.plot(hist_r, color='red', label='Canal Rojo (R)')

plt.title("Distribución de Intensidades por Canal de Color")
plt.xlabel("Valor del Píxel (0-255)")
plt.ylabel("Frecuencia (Cantidad de píxeles)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("histograma_resultado.png")