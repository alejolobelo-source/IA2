import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 Cargar la imagen en escala de grises
imagen = cv2.imread('ll.jpg', cv2.IMREAD_GRAYSCALE)

# 2 Generar tres tipos de bordes usando diferentes métodos
# Sobel X: detecta bordes verticales
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
# Sobel Y: detecta bordes horizontale
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
# Canny con umbrales por defecto: combina bordes y filtra
bordes_canny_base = cv2.Canny(imagen, 50, 150)
# Mostrar los resultados en un panel para comparar
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(sobel_x_abs, cmap='gray')
plt.title('Sobel X (Bordes Verticales)')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(sobel_y_abs, cmap='gray')
plt.title('Sobel Y (Bordes Horizontales)')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(bordes_canny_base, cmap='gray')
plt.title('Canny Base (50, 150)')
plt.axis('off')
plt.tight_layout()
plt.show()
# Probar cómo cambia el resultado con diferentes umbrales en Canny
# Bajos umbrales: captura más detalles, incluyendo ruido
canny_sensible = cv2.Canny(imagen, 10, 50)
# Altos umbrales: solo mantiene los bordes más fuertes
canny_estricto = cv2.Canny(imagen, 200, 250)
# Mostrar los tres resultados de Canny en una sola fila
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(canny_sensible, cmap='gray')
plt.title('Canny Sensible (10, 50)')
plt.subplot(1, 3, 2)
plt.imshow(bordes_canny_base, cmap='gray')
plt.title('Canny Base (50, 150)')
plt.subplot(1, 3, 3)
plt.imshow(canny_estricto, cmap='gray')
plt.title('Canny Estricto (200, 250)')
plt.savefig('resultado_bordes.png')