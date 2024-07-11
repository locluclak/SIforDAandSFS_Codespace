import numpy as np
from mpmath import mp

mp.dps = 1000


def f(z_interval, etajTy, mu, tn_sigma):
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        cdf_ar = mp.ncdf((ar - mu) / tn_sigma)
        cdf_al = mp.ncdf((al - mu) / tn_sigma)

        denominator = denominator + cdf_ar - cdf_al

        if etajTy >= ar:
            numerator = numerator + cdf_ar - cdf_al
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - mu) / tn_sigma) - cdf_al

    if denominator != 0:
        return float(numerator / denominator)
    else:
        return np.Inf


def find_root(z_interval, etajTy, tn_sigma, y, lb, ub):
    """
    searches for solution to f(x) = y in (lb, ub), where
    f is a monotone decreasing function
    """

    a, b = lb, ub
    fa, fb = f(z_interval, etajTy, a, tn_sigma), f(z_interval, etajTy, b, tn_sigma)

    if (fa > y) and (fb > y):
        while fb > y:
            b = b + ((b - a)/2)
            fb = f(z_interval, etajTy, b, tn_sigma)
            if fb == np.Inf:
                fb = 0

    elif (fa < y) and (fb < y):
        while fa < y:
            a = a - ((b - a)/2)
            fa = f(z_interval, etajTy, a, tn_sigma)
            if fa == np.Inf:
                fa = 1

    c = None

    while np.abs(b - a) > 1e-3:
        c = (a + b) / 2
        fc = f(z_interval, etajTy, c, tn_sigma)

        if np.around(fc, 4) == y:
            break

        if fc > y:
            a = c
        else:
            b = c

    return c


def equal_tailed_interval(z_interval, etajTy, alpha, tn_sigma):
    lb = -20
    ub = 20

    L = find_root(z_interval, etajTy, tn_sigma, 1.0 - 0.5 * alpha, lb, ub)
    U = find_root(z_interval, etajTy, tn_sigma, 0.5 * alpha, lb, ub)

    return np.array([L, U])


def compute_ci_with_constructed_interval(z_interval, etaj, etajTy, cov, alpha):
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    ci = equal_tailed_interval(z_interval, etajTy, alpha, tn_sigma)

    return ci