import matplotlib.pyplot as plt
import numpy as np

# Configurar plano de 0 a 10 con cuadrícula tipo cuaderno
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#FDFBF7')
ax.set_facecolor('#FDFBF7')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks(range(0, 11))
ax.set_yticks(range(0, 11))
ax.grid(True, color='#B0BEC5', linestyle='-', linewidth=0.8, alpha=0.7)

# Puntos de las clases
clase_a = np.array([[2, 2], [3, 3], [4, 2]])
clase_b = np.array([[6, 6], [7, 8], [8, 7]])

# Dibujar puntos (Clase A: círculos azules, Clase B: cruces verdes)
ax.scatter(clase_a[:, 0], clase_a[:, 1], color='#0D47A1', s=100, label='Clase A (Círculos)', marker='o', zorder=4)
ax.scatter(clase_b[:, 0], clase_b[:, 1], color='#1B5E20', s=120, label='Clase B (Equis)', marker='x', linewidths=2.5, zorder=4)

# Encerrar en rojo los Vectores de Soporte: (3,3), (4,2) y (6,6)
vs = np.array([[3, 3], [4, 2], [6, 6]])
ax.scatter(vs[:, 0], vs[:, 1], s=280, facecolors='none', edgecolors='#D32F2F', linewidths=2.5, label='Vectores de Soporte', zorder=5)

# Trazar hiperplano óptimo descendente: y = -x + 9
x_vals = np.linspace(0, 10, 100)
ax.plot(x_vals, -x_vals + 9, color='#111111', linewidth=2.5, label='Hiperplano: x + y = 9', zorder=3)

# Márgenes simétricos
ax.plot(x_vals, -x_vals + 6, color='#0D47A1', linestyle='--', linewidth=1.5, label='Margen Clase A (x + y = 6)', zorder=2)
ax.plot(x_vals, -x_vals + 12, color='#1B5E20', linestyle='--', linewidth=1.5, label='Margen Clase B (x + y = 12)', zorder=2)

ax.set_title('Taller Analítico: Plano Cartesiano SVM (Ejes 0 a 10)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Eje X', fontweight='bold')
ax.set_ylabel('Eje Y', fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9)

# Guardar la imagen directamente en tu carpeta
plt.savefig('grafica_svm_entrega.png', dpi=300, bbox_inches='tight')
print("--> Imagen generada con éxito como 'grafica_svm_entrega.png'")