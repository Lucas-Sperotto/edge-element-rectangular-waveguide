import numpy as np
from scipy.sparse.linalg import eigsh
from assemble_edge_elements import assemble_edge_matrices, check_zero_rows_cols


def apply_dirichlet_boundary(K, M, edge_dict, triangles, points=None, plot=False):
    from collections import defaultdict

    edge_count = defaultdict(int)

    # Conta quantos triângulos cada aresta pertence
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for e in edges:
            e_sorted = tuple(sorted(e))
            edge_count[e_sorted] += 1

    # Encontra as arestas de contorno
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    print(f"\n🔍 Total de arestas de contorno: {len(boundary_edges)}")
    for i, edge in enumerate(boundary_edges):
        idx = edge_dict.get(edge, None)
        if idx is None:
            print(f"❌ Aresta {edge} não encontrada no edge_dict!")
            continue
        #print(f"  {i+1:2d}: aresta {edge} (índice: {idx})")
        if points is not None:
            p1, p2 = points[edge[0]], points[edge[1]]
            #print(f"      ↪ coord: {p1} → {p2}")

        # Aplicar condição de contorno
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1.0
        
        M[idx, :] = 0
        M[:, idx] = 0
        M[idx, idx] = 1.0

    return K.tocsr(), M.tocsr()



def solve_waveguide_modes(points, triangles, num_modes=6, sigma=1e-3):
    """
    Resolve os modos do guia de onda (TE ou TM) com elementos de aresta.
    
    Parâmetros:
        points : ndarray
            Coordenadas dos nós da malha (N x 2)
        triangles : ndarray
            Conectividade dos triângulos (M x 3)
        num_modes : int
            Número de modos a resolver
        sigma : float
            Valor de deslocamento espectral para o solver

    Retorna:
        fc : array
            Frequências de corte (Hz)
        modos : list
            Lista de autovetores correspondentes
        lambdas : array
            Autovalores do problema
        edge_dict : dict
            Dicionário de mapeamento de arestas
    """
    from scipy.constants import c, pi

    print("Montando matrizes K e M com elementos de aresta...")
    K, M, edge_dict = assemble_edge_matrices(points, triangles)
    K, M = apply_dirichlet_boundary(K, M, edge_dict, triangles, points=points, plot=True)
    check_zero_rows_cols(K, "K")
    check_zero_rows_cols(M, "M")
    from numpy.linalg import svd

    s_K = svd(K.toarray(), compute_uv=False)
    s_M = svd(M.toarray(), compute_uv=False)

    cond_K = s_K[0] / s_K[-1]
    cond_M = s_M[0] / s_M[-1]

    print(f"Número de condição (SVD) de K: {cond_K:.2e}")
    print(f"Número de condição (SVD) de M: {cond_M:.2e}")
    print("Resolvendo problema de autovalores...")
    vals, vecs = eigsh(K, k=60, M=M, sigma=sigma, which='SM')
    vals = np.real(vals)

    kc = np.sqrt(np.clip(vals, 0, None))
    fc = c * kc / (2 * pi)

    modos = [vecs[:, i] for i in range(num_modes)]

    return kc, modos, vals, edge_dict
