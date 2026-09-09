import cv2
import matplotlib.pyplot as plt
import numpy as np

# =======================================================
# SESIÓN 3: SEGMENTACIÓN Y OPERACIONES MORFOLÓGICAS
# =======================================================
print("=== INICIANDO TALLER DE LABORATORIO: SESIÓN 3 ===")

# 1. Cargar imagen de prueba (usamos 'll.jpg' existente)
imagen = cv2.imread('ll.jpg', cv2.IMREAD_GRAYSCALE)

if imagen is None:
  print("ERROR: No se encontró la imagen 'll.jpg' en la carpeta.")
else:
  # 2. Binarización Estática (introduce ruido intencional) y Otsu
  T_estatico = 127
  _, img_binaria = cv2.threshold(
      imagen, T_estatico, 255, cv2.THRESH_BINARY
  )  #[cite: 3]
  T_otsu, img_otsu = cv2.threshold(
      imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
  )  #[cite: 3]

  print(f"Umbral Estático aplicado: T = {T_estatico}")
  print(f"Umbral Óptimo calculado por Otsu: T = {T_otsu:.2f}")  #[cite: 3]

  # 3. Elemento Estructurante (Kernel 3x3)
  kernel = np.ones((3, 3), np.uint8)  #[cite: 3]

  # 4. Operación de APERTURA (Erosión + Dilatación)
  # Elimina el ruido blanco fuera del objeto principal
  img_apertura = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, kernel)

  # 5. Operación de CIERRE (Dilatación + Erosión)
  # Rellena los huecos negros dentro del objeto principal
  img_cierre = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, kernel)

  # =======================================================
  # GENERACIÓN DE DASHBOARD COMPARATIVO
  # =======================================================
  plt.figure(figsize=(14, 8))
  plt.suptitle(
      "SEGMENTACIÓN MORFOLÓGICA Y LIMPIEZA DE RUIDO (SESIÓN 3)",
      fontsize=14,
      fontweight="bold",
  )

  plt.subplot(2, 2, 1)
  plt.imshow(img_binaria, cmap="gray")
  plt.title(f"1. Binarizada Estática (T={T_estatico})")
  plt.axis("off")

  plt.subplot(2, 2, 2)
  plt.imshow(img_otsu, cmap="gray")
  plt.title(f"2. Binarizada Otsu (T_óptimo={T_otsu:.1f})")
  plt.axis("off")

  plt.subplot(2, 2, 3)
  plt.imshow(img_apertura, cmap="gray")
  plt.title("3. Apertura (Erosión -> Dilatación)\n[Limpia ruido externo]")
  plt.axis("off")

  plt.subplot(2, 2, 4)
  plt.imshow(img_cierre, cmap="gray")
  plt.title("4. Cierre (Dilatación -> Erosión)\n[Rellena huecos internos]")
  plt.axis("off")

  plt.tight_layout()
  plt.savefig("resultado_segmentacion_sesion3.png")
  print("-> Dashboard generado como 'resultado_segmentacion_sesion3.png'")

  # Conclusión del análisis
  print("\n--- CONCLUSIÓN ---")
  print(
      "• La APERTURA fue más efectiva para eliminar píxeles aislados de ruido"
      " en el fondo."
  )
  print(
      "• El CIERRE funcionó mejor si el objetivo era sellar huecos y conectar"
      " regiones del objeto."
  )