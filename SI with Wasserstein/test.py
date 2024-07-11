import numpy as np
from scipy.optimize import linprog

def construct_omega(n, m):
    c = np.ones((m, 1))
    for _ in range(n - 1):
        c = np.hstack((c, np.zeros((m, 1))))
    c = np.hstack((c, - np.identity(m)))

    for _ in range(n - 1):
        c_temp = np.zeros((m, 1))
        for __ in range(n - 1):
            if _ == __:
                c_temp = np.hstack((c_temp, np.ones((m, 1))))
            else:
                c_temp = np.hstack((c_temp, np.zeros((m, 1))))
        c_temp = np.hstack((c_temp, - np.identity(m)))
        c = np.vstack((c, c_temp))

    return c

def construct_Sh(n, m):
    S = np.ones((1, m))
    
    for _ in range(n - 1):
        S = np.hstack((S, np.zeros((1, m))))
    
    for _ in range(n - 1):
        S_temp = np.zeros((1, m))
        for __ in range(n - 1):
            if _ == __:
                S_temp = np.hstack((S_temp, np.ones((1, m))))
            else:
                S_temp = np.hstack((S_temp, np.zeros((1, m))))
        S = np.vstack((S, S_temp))
        
    Mc = np.identity(m)
    for _ in range(n - 1):
        Mc = np.hstack((Mc, np.identity((m))))

    S = np.vstack((S, Mc))

    h = np.vstack((np.ones((n, 1)) / n, np.ones((m, 1)) / m))

    return S, h

def run():
    n = 7
    m = 5
    p = 1
    
    X = np.random.normal(loc = 0, scale = 1, size = (n, p))
    Y = np.random.normal(loc = 0, scale = 1, size = (m, p))

    # Construct cost vector
    omega = construct_omega(n, m)
    XY = np.vstack((X, Y))
    theta = np.sign(omega.dot(XY)) * omega
    c = theta.dot(XY)
    print(c.shape)

    # Construct S and h
    S, h = construct_Sh(n, m)
    A_eq = S[: -1]
    b_eq = h[: -1]


    # Linear Program
    res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = (0, None), method = 'simplex')
    print(res.basis)

if __name__ == '__main__':
    run()