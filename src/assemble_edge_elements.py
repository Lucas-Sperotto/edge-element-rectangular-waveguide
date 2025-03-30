import numpy as np
from scipy.sparse import lil_matrix

def check_zero_rows_cols(matrix, name=""):
    from scipy.sparse import csr_matrix
    matrix = matrix.tocsr()
    for i in range(matrix.shape[0]):
        if matrix[i].nnz == 0:
            print(f"⚠️ Linha/coluna {i} de {name} está totalmente zerada!")


def assemble_edge_matrices(points, triangles):
    """
    Monta as matrizes de rigidez (K) e massa (M) usando elementos de aresta (Whitney 1-form),
    com base no artigo de Reddy (1994, NASA TP-3485).
    """

    edge_dict = {}
    edge_id = 0
    tri_edges = []

    # Identificação única das arestas e mapeamento local-global
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        local_edge_ids = []
        for e in edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in edge_dict:
                edge_dict[e_sorted] = edge_id
                edge_id += 1
            local_edge_ids.append(edge_dict[e_sorted])
        tri_edges.append(local_edge_ids)

    Ne = edge_id
    K = lil_matrix((Ne, Ne))
    M = lil_matrix((Ne, Ne))

    for tri, edge_ids in zip(triangles, tri_edges):
        # Coordenadas dos vértices do triângulo
        n1, n2, n3 = tri
        v1, v2, v3 = points[n1], points[n2], points[n3]
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3

        # Centroide do triângulo
        Xmed = (x1 + x2 + x3) / 3
        Ymed = (y1 + y2 + y3) / 3

        # Área do triângulo (determinante da matriz da base)
        A = 0.5 * np.abs(np.linalg.det(np.array([[1, x1, y1],
                                                 [1, x2, y2],
                                                 [1, x3, y3]])))

        # Comprimentos das arestas
        edge_node_pairs = [(n1, n2), (n2, n3), (n3, n1)]
        L = [np.linalg.norm(points[nj] - points[ni]) for ni, nj in edge_node_pairs]

        # Índices dos nós (i, j, k) associados a cada aresta
        ijk = [(n1, n2, n3), (n2, n3, n1), (n3, n1, n2)]

        # Cálculo dos coeficientes a_i, b_i, c_i
        a_vals, b_vals, c_vals = [], [], []
        for i, j, k in ijk:
            #xi, yi = points[i]
            xj, yj = points[j]
            xk, yk = points[k]

            ai = xj * yk - xk * yj
            bi = yj - yk
            ci = xk - xj

            a_vals.append(ai)
            b_vals.append(bi)
            c_vals.append(ci)

        # Cálculo dos coeficientes Am, Bm, Cm, Dm
        Acoef, Bcoef, Ccoef, Dcoef = [], [], [], []
        for m in range(3):
            i = m
            j = (m + 1) % 3  # i → j → k circularmente

            Am = a_vals[i] * b_vals[j] - a_vals[j] * b_vals[i]
            Bm = c_vals[i] * b_vals[j] - c_vals[j] * b_vals[i]
            Cm = a_vals[i] * c_vals[j] - a_vals[j] * c_vals[i]
            Dm = b_vals[i] * c_vals[j] - b_vals[j] * c_vals[i]

            Acoef.append(Am)
            Bcoef.append(Bm)
            Ccoef.append(Cm)
            Dcoef.append(Dm)
        


        #print("Acoef:", Acoef)
        #print("Bcoef:", Bcoef)
        #print("Ccoef:", Ccoef)
        #print("Dcoef:", Dcoef)
        #print("Área:", A)

        # === Montagem da matriz de rigidez (K) ===
        for m in range(3):
            for n in range(3):
                k_mn = ((L[m] * L[n]) / (16 * A**3)) * (Dcoef[m] * Dcoef[n])
                if np.isclose(k_mn, 0.0, atol=1e-14):
                    print("kmn é numericamente zero:", Mmn, "para m =", m, "n =", n)
                K[edge_ids[m], edge_ids[n]] += k_mn

        # === Montagem da matriz de massa (M) ===
        xi_vals = [x1, x2, x3]
        yi_vals = [y1, y2, y3]

        sum_yi = sum([y**2 + 9 * Ymed**2 for y in yi_vals])
        sum_xi = sum([x**2 + 9 * Xmed**2 for x in xi_vals])

        for m in range(3):
            for n in range(3):
                It1 = Acoef[m] * Acoef[n] + Ccoef[m] * Ccoef[n]
                It2 = (Ccoef[m] * Dcoef[n] + Ccoef[n] * Dcoef[m]) * Xmed
                It3 = (Acoef[m] * Bcoef[n] + Acoef[n] * Bcoef[m]) * Ymed
                It4 = ((Bcoef[m] * Bcoef[n]) / 12) * sum_yi
                It5 = ((Dcoef[m] * Dcoef[n]) / 12) * sum_xi

                Mmn = ((L[m] * L[n]) / (16 * A**3)) * (It1 + It2 + It3 + It4 + It5)
                if np.isclose(Mmn, 0.0, atol=1e-14):
                    print("Mmn é numericamente zero:", Mmn, "para m =", m, "n =", n)
                M[edge_ids[m], edge_ids[n]] += Mmn

    return K, M, edge_dict
