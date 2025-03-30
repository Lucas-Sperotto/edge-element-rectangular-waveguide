import numpy as np
import meshio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from solve_modes import solve_waveguide_modes
from postprocess import plot_modes_tricontourf


# === Parâmetros do guia de onda ===
a = 1.0  # largura (m)
b = 0.5  # altura (m)
filename = "rect_edge"
num_modes = 24

# === Geração da malha via Gmsh ===
def generate_rectangle_mesh(a, b, filename):
    lc = min(a, b) / 20
    geo_content = f"""
    lc = {lc};
    Point(1) = {{0, 0, 0, lc}};
    Point(2) = {{{a}, 0, 0, lc}};
    Point(3) = {{{a}, {b}, 0, lc}};
    Point(4) = {{0, {b}, 0, lc}};
    Line(1) = {{1, 2}};
    Line(2) = {{2, 3}};
    Line(3) = {{3, 4}};
    Line(4) = {{4, 1}};
    Line Loop(5) = {{1, 2, 3, 4}};
    Plane Surface(6) = {{5}};
    Physical Surface(200) = {{6}};
    Mesh 2;
    Save "{filename}.msh";
    """
    with open(f"{filename}.geo", "w") as f:
        f.write(geo_content)
    os.system(f"gmsh {filename}.geo -2 -format msh2")

# === Executa ===
if __name__ == "__main__":
    print("Gerando malha...")
    generate_rectangle_mesh(a, b, filename)
    
    print("Lendo malha...")
    mesh = meshio.read(f"{filename}.msh")
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]

    print("Resolvendo modos...")
    kc, modos, lambdas, edge_dict = solve_waveguide_modes(points, triangles, num_modes=num_modes, sigma=0)
    kc_a = kc * a
    print("\n=== kc·a (adimensional) — apenas valores positivos ===")
    for i, f in enumerate(kc_a):
        #if f > 1e-12:
        print(f"Modo {i+1:2d}: {f:.4e}")

modos_validos = []
kc_a_validos = []
for i, m in enumerate(modos):
    if np.max(np.abs(m)) > 1e-8 and kc_a[i] > 1e6:
        modos_validos.append(m)
        kc_a_validos.append(kc_a[i])

print(f"\nModos válidos encontrados: {len(modos_validos)}")
plot_modes_tricontourf(points, triangles, modos_validos, edge_dict, filename='te_tm_edge')