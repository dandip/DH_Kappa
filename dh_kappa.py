###############################################################################
# General Information
###############################################################################
# General Description : Computes the DiPietro-Hazari Kappa metric
# Last Edited : 9/17/2022

###############################################################################
# Dependencies
###############################################################################

import numpy as np
import operator as op
from functools import reduce

###############################################################################
# Function
###############################################################################

def dh_kappa(A, B):
    """Computes DH Kappa for annotation matrix A and label matrix B

    Suppose you have n pieces of data d_1, ... , d_n with m possible categories
    c_1, ... , c_m. Each piece of data is assessed by N annotators; there are n
    proposed labels as well, denoted l_1, ... , l_n, where each l_i denotes the
    proposed category label of d_i. Let A define the n x m matrix where a_ij
    indicates the number of annotators that placed d_i in c_j. Let B denote the
    n x m matrix where b_ij = 1 if l_i = c_j and 0 otherwise. Using these
    matrices A and B, this function computes the DiPietro-Hazari Kappa metric.

    ~ Parameters ~
    - A (n x m) numpy matrix
    - B (n x m) numpy matrix

    ~ Returns (float) ~ : DH Kappa value
    """
    def col_sum(mat):
        return np.sum(mat, 0)
    def row_sum(mat):
        return np.sum(mat, 1)

    def ncr(n, r):
        if (n == 0 and r != 0):
            return 0
        elif (n == 0 and r == 0):
            return 1
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom
    def choose_two_map(num):
        return ncr(num, 2)
    f = np.vectorize(choose_two_map)

    eps = 0.00000000001

    N = np.sum(A[0])
    n = A.shape[0]
    m = A.shape[1]

    C = col_sum(A) * (1 / (N * n))
    L = col_sum(B) * (1 / n)
    C_E = row_sum(np.multiply(np.multiply(C, C), L).reshape(1, -1))

    stacked_C = np.vstack(list(C for i in range(m)))
    T = np.multiply(stacked_C, np.ones((m,m)) - np.eye(m))
    C_F = col_sum(np.multiply(row_sum(np.multiply(T, T)), L.T)).item()

    R = (col_sum(row_sum(np.multiply(f(A), B)) * (1 / ncr(N, 2))) * (1 / n)).item()

    S = (col_sum(row_sum(np.multiply(f(A), np.ones((n,m)) - B)) * (1 / ncr(N, 2))) * (1 / n)).item()

    K_dh = ((R - S) - (C_E - C_F)) / (1 - (C_E - C_F) + eps)
    return K_dh.item()
