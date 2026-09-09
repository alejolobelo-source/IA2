import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

print("--- INICIANDO PIPELINE DE INTEGRACIÓN (TALLER 1 + TALLER 2) ---")

# Cambia el nombre al archivo de imagen que tengas en tu carpeta
nombre_imagen = 'll.jpg'

# 1. VERIFICACIÓN DE ARCHIVO
if not os.path.exists(nombre_imagen):
  print(
      f"ERROR: No se encontró la imagen '{nombre_imagen}'. Asegúrate de colocarla en la misma carpeta."
  )
else:
  # 2. CARGA DE IMAGEN (TALLER 2)
  img_bgr = cv2.imread(nombre_imagen)
  print("1. Imagen original cargada correctamente.")

  # 3. TRANSFORMACIÓN ESPACIAL A ESCALA DE GRISES PONDERADA (TALLER 2)
  B = img_bgr[:, :, 0]
  G = img_bgr[:, :, 1]
  R = img_bgr[:, :, 2]

  # Ecuación de luminancia CIE / ITU-R BT.601
  img_gris = (0.114 * B) + (0.587 * G) + (0.299 * R)
  img_gris = np.clip(img_gris, 0, 255).astype(np.uint8)
  print("2. Imagen transformada a Escala de Grises Ponderada.")

  # 4. TRANSFORMACIÓN AFÍN: BRILLO Y CONTRASTE (TALLER 1)
  alpha = 0.5  # Escalar de contraste (50%)
  beta = -50.0  # Escalar de brillo (reducción de -50)

  img_oscura = (alpha * img_gris) + beta
  img_oscura = np.clip(img_oscura, 0, 255).astype(np.uint8)
  print(
      "3. Imagen procesada mediante Transformación Afín (Ajuste de brillo y"
      " contraste)."
  )

  # 5. DETECCIÓN DE TEXTURAS / BORDES MEDIANTE CONVOLUCIÓN (TALLER 1)
  Kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

  img_bordes = cv2.filter2D(img_gris, cv2.CV_8U, Kernel)
  print("4. Convolución de Kernel aplicada (Detección de bordes).")

  # 6. ANÁLISIS ESTADÍSTICO RESULTANTE (TALLER 2)
  hist_oscuro = cv2.calcHist([img_oscura], [0], None, [256], [0, 256])
  print("5. Procesamiento finalizado. Generando visualización...")

  # ==============================================================================
  # RENDERIZADO DEL DASHBOARD VISUAL (MATPLOTLIB)
  # ==============================================================================
  plt.figure(figsize=(14, 10))
  plt.suptitle(
      "PIPELINE DE INTEGRACIÓN: ÁLGEBRA LINEAL Y VISIÓN ARTIFICIAL",
      fontsize=16,
      fontweight="bold",
  )

  # Cuadro 1: Imagen Original BGR convertida a RGB para renderizado correcto
  plt.subplot(2, 2, 1)
  plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
  plt.title("1. Imagen Original (Extracción de Tensores)")
  plt.axis("off")

  # Cuadro 2: Escala de Grises Ponderada
  plt.subplot(2, 2, 2)
  plt.imshow(img_gris, cmap="gray")
  plt.title("2. Escala de Grises (Matemática Ponderada)")
  plt.axis("off")

  # Cuadro 3: Transformación Afín
  plt.subplot(2, 2, 3)
  plt.imshow(img_oscura, cmap="gray", vmin=0, vmax=255)
  plt.title("3. Transformación Afín (Brillo -50, Contraste 50%)")
  plt.axis("off")

  # Cuadro 4: Detección de Bordes
  plt.subplot(2, 2, 4)
  plt.imshow(img_bordes, cmap="gray")
  plt.title("4. Detección de Bordes (Producto Hadamard con Kernel)")
  plt.axis("off")

  plt.tight_layout()

  # Guardar la imagen del resultado en la misma carpeta
  plt.savefig("resultado_integrador.png")
  print("-> Dashboard guardado como 'resultado_integrador.png'")