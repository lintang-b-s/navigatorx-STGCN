import numpy as np 
import pandas as pd
import os 

def chebyshev_polynomial_approx(L, K, n_vertex):
    L0, L1 = np.asmatrix(np.identity(n_vertex)), np.asmatrix(np.copy(L))
    L_list = [L0, L1]
    for k in range(K-2):
        Lk = np.asmatrix(2*L*L1 - L0) 
        L_list.append(np.copy(Lk))   
        L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Lk))
    return np.concatenate(L_list, axis=-1) # (n_vertex, n_vertex*K)

def scaled_normalized_laplacian(W):
    n_vertex = W.shape[0]
    d = np.sum(W, axis=1)
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n_vertex):
        for j in range(n_vertex):
            if (d[i]>0) and (d[j]>0):
                L[i,j] = L[i,j] / np.sqrt(d[i]*d[j])
    lambda_max = np.linalg.eigvals(L).real.max()
    return np.asmatrix(2*L/lambda_max - np.identity(n_vertex))

def build_graph_kernel(dataset_name, dataset_file_name, K):
    W = pd.read_csv(os.path.join(dataset_name, dataset_file_name), header=None).values
    n = W.shape[0]
    W = W / 10000.
    Wsquared, W_mask = W*W, np.ones([n,n])-np.identity(n)
   
    W = np.exp(-Wsquared / 0.1) * (np.exp(-Wsquared / 0.1) >= 0.5) * W_mask #  equation (0.1)

    L = scaled_normalized_laplacian(W)
    Lk = chebyshev_polynomial_approx(L, K, n)
    return Lk
