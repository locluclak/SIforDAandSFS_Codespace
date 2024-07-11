import numpy as np 

def generate(m, n, true_beta):
    """return data X, Y"""
    X = np.random.rand(m, n)
    # Generate random noise (error term)
    epsilon = np.random.normal(0, 1, m).reshape((-1,1))  # Normally distributed noise

    # Compute the target variable Y
    true_beta = true_beta.reshape((-1,1))
    
    Y = np.dot(X, true_beta) + epsilon
    return X, Y