import cv2
import matplotlib.pyplot as plt
import numpy as np

# =======================================================
# SESIÓN 4: TALLER DE LABORATORIO - ESTRATEGIAS DE SUAVIZADO
# =======================================================
print("=== INICIANDO LABORATORIO: ESTRATEGIAS DE SUAVIZADO ===")

# 1. Cargar imagen base
imagen = cv2.imread('ll.jpg')

if imagen is None:
  print("ERROR: No se encontró la imagen 'll.jpg'. Asegúrate de estar en 'Seccion 4'.")
else:
  # 1. Crear imagen con RUIDO DE SAL Y PIMIENTA (Punto 1)
  imagen_ruidosa = imagen.copy()
  probabilidad = 0.08  # 8% de píxeles afectados por ruido

  # Añadir Ruido de Sal (blanco = 255)
  num_sal = np.ceil(probabilidad * imagen.size * 0.5)
  coords_sal = [
      np.random.randint(0, i - 1, int(num_sal)) for i in imagen.shape[:2]
  ]
  imagen_ruidosa[tuple(coords_sal)] = 255

  # Añadir Ruido de Pimienta (negro = 0)
  num_pimienta = np.ceil(probabilidad * imagen.size * 0.5)
  coords_pimienta = [
      np.random.randint(0, i - 1, int(num_pimienta)) for i in imagen.shape[:2]
  ]
  imagen_ruidosa[tuple(coords_pimienta)] = 0

  # 2. Aplicar Filtros con Kernel agresivo 7x7 (Punto 2)
  kernel_size = 7

  blur_media = cv2.blur(imagen_ruidosa, (kernel_size, kernel_size))  #
  blur_gauss = cv2.GaussianBlur(
      imagen_ruidosa, (kernel_size, kernel_size), 0
  )  #
  blur_mediana = cv2.medianBlur(imagen_ruidosa, kernel_size)  #

  # 3. Guardar y visualizar resultados en Dashboard (Punto 3 - Adaptado para Codespaces)
  plt.figure(figsize=(12, 10))
  plt.suptitle(
      "LABORATORIO SESIÓN 4: COMPARATIVA DE FILTRADO ESPACIAL (KERNEL 7x7)",
      fontsize=14,
      fontweight="bold",
  )

  plt.subplot(2, 2, 1)
  plt.imshow(cv2.cvtColor(imagen_ruidosa, cv2.COLOR_BGR2RGB))
  plt.title("1. Con Ruido Sal y Pimienta")
  plt.axis("off")

  plt.subplot(2, 2, 2)
  plt.imshow(cv2.cvtColor(blur_media, cv2.COLOR_BGR2RGB))
  plt.title("2. Filtro de Media (Promedio 7x7)\n[Genera manchas grises]")
  plt.axis("off")

  plt.subplot(2, 2, 3)
  plt.imshow(cv2.cvtColor(blur_gauss, cv2.COLOR_BGR2RGB))
  plt.title("3. Filtro Gaussiano (7x7)\n[Suavizado borroso]")
  plt.axis("off")

  plt.subplot(2, 2, 4)
  plt.imshow(cv2.cvtColor(blur_mediana, cv2.COLOR_BGR2RGB))
  plt.title("4. Filtro de Mediana (7x7)\n[Elimina ruido limpiamente]")
  plt.axis("off")

  plt.tight_layout()
  plt.savefig("resultado_laboratorio_suavizado.png")
  print(
      "-> Dashboard guardado exitosamente como 'resultado_laboratorio_suavizado.png'"
  )