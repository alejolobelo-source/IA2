import cv2
import numpy as np

# 1. Cargar la imagen (Asegúrate de subir 'monedas.jpg' a la carpeta Seccion 5)[cite: 2]
imagen_color = cv2.imread('monedas.jpg')
imagen_copia = imagen_color.copy()

# 2. Aplicación del Pipeline mejorado
# Conversión a grises
gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Desenfoque más fuerte para difuminar los brillos metálicos
blur = cv2.GaussianBlur(gris, (21, 21), 0)

# Umbralización con Otsu
_, umbral = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Kernel más grande y doble iteración para sellar por completo los contornos
kernel = np.ones((15, 15), np.uint8)
limpia = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel, iterations=2)

# Detección de Contornos
contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lógica empresarial: Definimos el valor "X" (ajusta este valor según la resolución de tu imagen)[cite: 2]
umbral_area_grande = 6000  # Ajusta este número según lo que leas en tu terminal

print("--- Áreas de los objetos detectados ---")

for i, cnt in enumerate(contornos):
    # 3. Imprimir en consola el área de cada objeto encontrado[cite: 2]
    area = cv2.contourArea(cnt)
    
    # Filtramos píxeles basura muy pequeños para no contarlos como objetos
    if area > 100:
        print(f"Objeto {i+1}: {area} píxeles")
        
        # Generar coordenadas del Bounding Box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 4. Lógica empresarial de clasificación[cite: 2]
        if area > umbral_area_grande:
            # Bounding Box Azul para objeto grande (OpenCV usa formato BGR: Blue, Green, Red)[cite: 2]
            color = (255, 0, 0)
        else:
            # Bounding Box Rojo para objeto pequeño[cite: 2]
            color = (0, 0, 255)
            
        cv2.rectangle(imagen_copia, (x, y), (x+w, y+h), color, 3)

# Guardar la imagen en lugar de usar imshow para visualizarla en Codespaces
cv2.imwrite('resultado_clasificador.jpg', imagen_copia)
print("Proceso terminado. Revisa el archivo 'resultado_clasificador.jpg'.")