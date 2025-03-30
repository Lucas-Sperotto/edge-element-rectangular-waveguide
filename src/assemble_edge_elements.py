import numpy as np
from scipy.sparse import lil_matrix

def check_zero_rows_cols(matrix, name=""):
    from scipy.sparse import csr_matrix
    matrix = matrix.tocsr()
    for i in range(matrix.shape[0]):
        if matrix[i].nnz == 0:
            print(f"⚠️ Linha/coluna {i} de {name} está totalmente zerada!")


from scipy.sparse import lil_matrix
import numpy as np

def assemble_edge_matrices(points, triangles):
    """
    Monta as matrizes de rigidez (K) e massa (M) usando elementos de aresta (Whitney 1-forms),
    com consideração da orientação local das arestas.
    """

    edge_dict = {}
    edge_id = 0
    tri_edges = []
    tri_orientations = []

    # Mapeamento global das arestas
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        local_edge_ids = []
        local_orientations = []

        for e in edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in edge_dict:
                edge_dict[e_sorted] = edge_id
                edge_id += 1
            local_edge_ids.append(edge_dict[e_sorted])
            orientation = 1 if e == e_sorted else -1
            local_orientations.append(orientation)

        tri_edges.append(local_edge_ids)
        tri_orientations.append(local_orientations)

    Ne = edge_id
    K = lil_matrix((Ne, Ne))
    M = lil_matrix((Ne, Ne))

    for tri, edge_ids, orientations in zip(triangles, tri_edges, tri_orientations):
        n1, n2, n3 = tri
        v1, v2, v3 = points[n1], points[n2], points[n3]
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3

        Xmed = (x1 + x2 + x3) / 3
        Ymed = (y1 + y2 + y3) / 3

        A = 0.5 * np.abs(np.linalg.det(np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ])))

        edge_node_pairs = [(n1, n2), (n2, n3), (n3, n1)]
        L = [np.linalg.norm(points[nj] - points[ni]) for ni, nj in edge_node_pairs]

        ijk = [(n1, n2, n3), (n2, n3, n1), (n3, n1, n2)]
        a_vals, b_vals, c_vals = [], [], []
        for i, j, k in ijk:
            xj, yj = points[j]
            xk, yk = points[k]
            ai = xj * yk - xk * yj
            bi = yj - yk
            ci = xk - xj
            a_vals.append(ai)
            b_vals.append(bi)
            c_vals.append(ci)

        Acoef, Bcoef, Ccoef, Dcoef = [], [], [], []
        for m in range(3):
            i = m
            j = (m + 1) % 3
            Am = a_vals[i] * b_vals[j] - a_vals[j] * b_vals[i]
            Bm = c_vals[i] * b_vals[j] - c_vals[j] * b_vals[i]
            Cm = a_vals[i] * c_vals[j] - a_vals[j] * c_vals[i]
            Dm = b_vals[i] * c_vals[j] - b_vals[j] * c_vals[i]
            Acoef.append(Am)
            Bcoef.append(Bm)
            Ccoef.append(Cm)
            Dcoef.append(Dm)

        xi_vals = [x1, x2, x3]
        yi_vals = [y1, y2, y3]
        sum_yi = sum([y**2 + 9 * Ymed**2 for y in yi_vals])
        sum_xi = sum([x**2 + 9 * Xmed**2 for x in xi_vals])

        # Montagem de K e M com orientação
        for m in range(3):
            for n in range(3):
                sign_m = orientations[m]
                sign_n = orientations[n]

                k_mn = ((L[m] * L[n]) / (4 * A**3)) * (Dcoef[m] * Dcoef[n])
                K[edge_ids[m], edge_ids[n]] += sign_m * sign_n * k_mn

                It1 = Acoef[m] * Acoef[n] + Ccoef[m] * Ccoef[n]
                It2 = (Ccoef[m] * Dcoef[n] + Ccoef[n] * Dcoef[m]) * Xmed
                It3 = (Acoef[m] * Bcoef[n] + Acoef[n] * Bcoef[m]) * Ymed
                It4 = ((Bcoef[m] * Bcoef[n]) / 12) * sum_yi
                It5 = ((Dcoef[m] * Dcoef[n]) / 12) * sum_xi

                Mmn = ((L[m] * L[n]) / (16 * A**3)) * (It1 + It2 + It3 + It4 + It5)
                M[edge_ids[m], edge_ids[n]] += sign_m * sign_n * Mmn

    return K, M, edge_dict

