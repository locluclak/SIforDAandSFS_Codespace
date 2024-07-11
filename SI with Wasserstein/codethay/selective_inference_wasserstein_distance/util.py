import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from mpmath import mp

mp.dps = 500


def plot(list_true, list_lb, list_ub):
    y = list_true

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(14, 5.2))

    lower_bound_A = list_lb

    upper_bound_A = list_ub

    lower_error_A = (np.array(y)) - np.array(lower_bound_A)
    upper_error_A = (np.array(upper_bound_A) - np.array(y))

    xi = np.arange(1, len(y) + 1, 1)
    plt.errorbar(xi, y, yerr=[lower_error_A, upper_error_A], fmt='o')

    plt.xlabel("Trial")
    plt.ylabel("CI")
    plt.tight_layout()
    plt.savefig('./results/ci_demonstration.png')
    plt.show()


def LP_solver(c_vec, S, u, G, h):
    # res = linprog(c_vec, A_ub=G, b_ub=h, A_eq=S, b_eq=u, method='simplex')
    res = linprog(c_vec, A_ub=G, b_ub=h, A_eq=S, b_eq=u, method='simplex',
                  options={'maxiter': 10000})
    return res


def construct_S_u_G_h(n, m):
    dim_t = n * m

    M_r = np.zeros((n, dim_t))

    for i in range(n):
        M_r[i, i * m:i * m + m] = np.ones(m)

    M_c = np.zeros((m, dim_t))

    for i in range(m):
        for j in range(i, dim_t, m):
            M_c[i, j] = 1.0

    S = np.vstack((M_r, M_c))
    u = np.vstack((np.ones((n, 1))/n, np.ones((m, 1))/m))

    # Remove any redundant row (e.g., last row)
    S = S[:-1, :]
    u = u[:-1, :]

    # Construct G
    G = -np.identity(dim_t)

    # Construct h
    h = np.zeros((dim_t, 1))

    return S, u, G, h


def construct_set_basis_non_basis_idx(t_hat):
    A = []
    Ac = []

    for i in range(len(t_hat)):
        if t_hat[i] != 0.0:
            A.append(i)
        else:
            Ac.append(i)

    return A, Ac


def construct_Theta(n, m, d, data_obs):
    idx_matrix = np.identity(n)
    Omega = None
    for i in range(n):
        temp_vec = None
        for j in range(n):
            if idx_matrix[i][j] == 1.0:
                if j == 0:
                    temp_vec = np.ones((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.ones((m, 1))))
            else:
                if j == 0:
                    temp_vec = np.zeros((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.zeros((m, 1))))

        temp_vec = np.hstack((temp_vec, -np.identity(m)))

        if i == 0:
            Omega = temp_vec.copy()
        else:
            Omega = np.vstack((Omega, temp_vec))

    Theta = np.zeros((n * m, n * d + m * d))

    list_sign = []
    list_kronecker_product = []

    for k in range(d):
        e_d_k = np.zeros((d, 1))
        e_d_k[k][0] = 1.0

        kronecker_product = np.kron(Omega, e_d_k.T)
        dot_product = np.dot(kronecker_product, data_obs)
        s_k = np.sign(dot_product)

        Theta = Theta + s_k * kronecker_product

        list_sign.append(s_k)
        list_kronecker_product.append(kronecker_product)

    return Theta, list_sign, list_kronecker_product


def compute_a_b(data, eta, dim_data):
    sq_norm = (np.linalg.norm(eta))**2

    e1 = np.identity(dim_data) - (np.dot(eta, eta.T))/sq_norm
    a = np.dot(e1, data)

    b = eta/sq_norm

    return a, b


def pivot_with_constructed_interval(z_interval, eta, etaTy, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(eta.T, cov), eta))[0][0]
    # print(tn_sigma)
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etaTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etaTy >= al) and (etaTy < ar):
            numerator = numerator + mp.ncdf((etaTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None