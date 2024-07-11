import numpy as np

import gen_data
import util
import ci


def run():
    n = 5
    m = 5
    d = 2
    delta_mu = 4
    dim_t = n * m

    X_obs, Y_obs, true_X, true_Y = gen_data.generate(n, m, d, delta_mu)

    X_obs_vec = X_obs.flatten().copy().reshape((n * d, 1))
    Y_obs_vec = Y_obs.flatten().copy().reshape((m * d, 1))

    true_X_vec = true_X.flatten().copy().reshape((n * d, 1))
    true_Y_vec = true_Y.flatten().copy().reshape((m * d, 1))

    data_obs = np.vstack((X_obs_vec, Y_obs_vec)).copy()

    # Cost matrix
    C = np.zeros((n, m))

    for i in range(n):
        e_x = X_obs[i, :]
        for j in range(m):
            e_y = Y_obs[j, :]
            C[i, j] = np.sum(np.abs(e_x - e_y))

    c_vec = C.copy().flatten().reshape((dim_t, 1))

    Theta, list_sign, list_kronecker_product = util.construct_Theta(n, m, d, data_obs)

    # LP
    S, u, G, h = util.construct_S_u_G_h(n, m)
    lp_res = util.LP_solver(c_vec, S, u, G, h)

    # OT plan vector
    t_hat = np.around(lp_res.x, 10)

    # Active set and non-active set
    A = lp_res.basis.copy().tolist()
    Ac = []
    for i in range(dim_t):
        if i not in A:
            Ac.append(i)

    t_hat_A = t_hat[A]

    # Construct eta
    Theta_A_ast = Theta[A, :].copy()
    eta = np.dot(Theta_A_ast.T, np.reshape(t_hat_A, (len(A), 1)))

    # etaTdata
    etaTdata = np.dot(eta.T, data_obs)[0][0]

    # Construct a_line and b_line
    a, b = util.compute_a_b(data_obs, eta, n * d + m * d)

    # Find interval
    q0 = np.dot(Theta, a)
    q1 = np.dot(Theta, b)

    q0_A = q0[A, :].copy()
    q0_Ac = q0[Ac, :].copy()

    q1_A = q1[A, :].copy()
    q1_Ac = q1[Ac, :].copy()

    S_ast_A = S[:, A].copy()
    S_ast_A_inv = np.linalg.inv(S_ast_A)
    S_ast_Ac = S[:, Ac].copy()

    v0 = q0_Ac.T - np.dot(q0_A.T, np.dot(S_ast_A_inv, S_ast_Ac))
    v1 = q1_Ac.T - np.dot(q1_A.T, np.dot(S_ast_A_inv, S_ast_Ac))

    v0 = np.around(v0.flatten(), 10)
    v1 = np.around(v1.flatten(), 10)

    Vminus = np.NINF
    Vplus = np.Inf

    # Selection event for A
    for i in range(len(v0)):
        a_coef = - v1[i]
        b_coef = v0[i]

        if a_coef == 0.0:
            continue

        temp = b_coef / a_coef

        if a_coef > 0:
            Vplus = min(Vplus, temp)
        elif a_coef < 0:
            Vminus = max(Vminus, temp)

    # Selection even for absolute in cost

    for k in range(d):
        element_wise_product = list_sign[k] * list_kronecker_product[k]

        vector_left = - np.dot(element_wise_product, b).copy().flatten()
        vector_right = np.dot(element_wise_product, a).flatten()

        for i in range(len(vector_left)):
            a_coef = vector_left[i]
            b_coef = vector_right[i]

            if -1e-8 <= a_coef <= 1e-8:
                if b_coef > 0:
                    continue

            temp = b_coef / a_coef

            if a_coef > 0:
                Vplus = min(Vplus, temp)
            elif a_coef < 0:
                Vminus = max(Vminus, temp)

    true_data = np.vstack((true_X_vec, true_Y_vec)).copy()
    tn_mu = np.dot(eta.T, true_data)[0][0]
    cov = np.identity(n * d + m * d)

    CI = ci.compute_ci_with_constructed_interval([[Vminus, Vplus]], eta, etaTdata, cov, 0.05)

    print()
    print('Estimated Distance:', etaTdata)
    print('True Distance:', tn_mu)
    print('Confidence Interval:', CI)
    print()


if __name__ == '__main__':
    np.random.seed(1)
    run()