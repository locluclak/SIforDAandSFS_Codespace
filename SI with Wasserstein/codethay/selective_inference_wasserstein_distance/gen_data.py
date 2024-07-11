import numpy as np
import matplotlib.pyplot as plt


def generate(n, m, d, delta_mu):
    mu_X = np.ones(d)
    mu_Y = mu_X + delta_mu

    true_X = np.array([mu_X for _ in range(n)])
    true_Y = np.array([mu_Y for _ in range(m)])

    X_obs = true_X + np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    Y_obs = true_Y + np.random.multivariate_normal(np.zeros(d), np.identity(d), m)

    return X_obs, Y_obs, true_X, true_Y


def generate_data_for_demo(n, m, d, delta_mu):
    true_X = np.array([[0], [1], [2], [3]])
    true_Y = true_X + delta_mu

    # print(true_X)
    # print(true_Y)

    X_obs = true_X + np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    Y_obs = true_Y + np.random.multivariate_normal(np.zeros(d), np.identity(d), m)

    return X_obs, Y_obs, true_X, true_Y


def generate_robust(n, m, d, delta_mu):
    mu_X = np.ones(d)
    mu_Y = mu_X + delta_mu

    true_X = np.array([mu_X for _ in range(n)])
    true_Y = np.array([mu_Y for _ in range(m)])

    # eps_X = np.random.laplace(0, 1, (n, 1))
    # eps_Y = np.random.laplace(0, 1, (m, 1))
    # eps_X = eps_X / np.sqrt(2)
    # eps_Y = eps_Y / np.sqrt(2)

    # from scipy.stats import skewnorm
    # eps_X = skewnorm.rvs(a=10, loc=0, scale=1, size=(n, 1))
    # eps_Y = skewnorm.rvs(a=10, loc=0, scale=1, size=(m, 1))
    # mean = skewnorm.mean(a=10, loc=0, scale=1)
    # std = skewnorm.std(a=10, loc=0, scale=1)
    # eps_X = (eps_X - mean) / std
    # eps_Y = (eps_Y - mean) / std

    # from scipy.stats import t
    # eps_X = t.rvs(df=20, loc=0, scale=1, size=(n, 1))
    # eps_Y = t.rvs(df=20, loc=0, scale=1, size=(m, 1))
    # mean = t.mean(df=20, loc=0, scale=1)
    # std = t.std(df=20, loc=0, scale=1)
    # eps_X = (eps_X - mean) / std
    # eps_Y = (eps_Y - mean) / std

    eps_X = np.random.normal(0, 1, (n, 1))
    eps_Y = np.random.normal(0, 1, (m, 1))
    std = np.std(np.random.normal(0, 1, 100), ddof=1)
    eps_X = eps_X / std
    eps_Y = eps_Y / std

    X_obs = true_X + eps_X
    Y_obs = true_Y + eps_Y

    return X_obs, Y_obs, true_X, true_Y