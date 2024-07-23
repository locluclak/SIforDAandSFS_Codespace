import numpy as np 
from sklearn import preprocessing
def generate(m, n, true_beta):
    """return data X, Y"""
    X = np.random.rand(m, n)
    # print(X)
    # # Normalize 
    # for row in X.T:
    #     row /= np.linalg.norm(row)
    # X = preprocessing.normalize(X)
    # Generate random noise (error term)
    epsilon = np.random.normal(0, 1, m).reshape((-1,1))  # Normally distributed noise

    # Compute the target variable Y
    true_beta = true_beta.reshape((-1,1))
    
    Y = np.dot(X, true_beta) + epsilon
    return X, Y

if __name__ == "__main__":
    x, y = generate(4,3,np.array([0,0,0]))
    print(x)

    print("______________")
    print(y)