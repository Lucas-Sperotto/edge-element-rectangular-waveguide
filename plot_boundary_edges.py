import meshio
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

# === Carrega malha ===
filename = "rect_edge"
mesh = meshio.read(f"{filename}.msh")
points = mesh.points[:, :2]
triangles = mesh.cells_dict["triangle"]

# === Identifica arestas ===
edge_count = defaultdict(int)
for tri in triangles:
    edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
    for e in edges:
        edge = tuple(sorted(e))
        edge_count[edge] += 1

boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

# === Cria pasta de saída ===
os.makedirs("out/img", exist_ok=True)

# === Plota a malha + arestas de contorno ===
plt.figure(figsize=(6, 4))

# Desenha todos os triângulos da malha
for tri in triangles:
    tri_pts = points[list(tri) + [tri[0]]]  # fecha o triângulo
    plt.plot(tri_pts[:, 0], tri_pts[:, 1], color='lightgray', lw=0.5)

# Desenha arestas de contorno em vermelho
for edge in boundary_edges:
    p1, p2 = points[edge[0]], points[edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=2)

# Configurações do gráfico
plt.plot(points[:, 0], points[:, 1], 'k.', markersize=3)
plt.axis('equal')
plt.title("Arestas de Contorno Detectadas")
plt.tight_layout()
plt.savefig("out/img/boundary_edges_all.png", dpi=300)
plt.close()
print("✅ Figura salva em: out/img/boundary_edges_all.png")
