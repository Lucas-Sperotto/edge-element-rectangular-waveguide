import numpy as np
import matplotlib.pyplot as plt
import os

def plot_modes_tricontourf(points, triangles, modos, edge_dict, filename='modo', save_path='out/img/', cmap='jet'):
    """
    Gera e salva os gráficos de cada modo usando tricontourf.
    Os valores dos modos estão nas arestas, então são interpolados para os nós.
    """
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    os.makedirs(save_path, exist_ok=True)

    # Mapeia valores de arestas para cada nó por média simples
    Ne = len(edge_dict)
    Nn = len(points)
    edge_to_nodes = [[] for _ in range(Nn)]
    edge_values_all = []

    # Mapeia cada aresta para os dois nós
    inverse_edge_dict = {v: k for k, v in edge_dict.items()}

    for modo in modos:
        node_vals = np.zeros(Nn)
        count = np.zeros(Nn)

        for edge_id, val in enumerate(modo):
            n1, n2 = inverse_edge_dict[edge_id]
            node_vals[n1] += val
            node_vals[n2] += val
            count[n1] += 1
            count[n2] += 1

        # Média dos valores por nó
        count[count == 0] = 1  # evitar divisão por zero
        node_vals /= count
        edge_values_all.append(node_vals)

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    for i, node_vals in enumerate(edge_values_all):
        plt.figure(figsize=(6, 5))
        plt.tricontourf(triang, node_vals, levels=100, cmap=cmap)
        plt.colorbar(label='Amplitude')
        plt.title(f'Modo {i+1}')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{filename}_modo_{i+1}.png"), dpi=300)
        plt.close()


