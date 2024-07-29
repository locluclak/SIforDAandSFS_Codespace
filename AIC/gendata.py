import numpy as np 
# from mpmath import mp
# mp.dps = 500
# from sklearn import preprocessing
def generate(m, n, true_beta):
    """return data X, Y"""
    X = np.random.rand(m, n)
    # print("X:", X)
    # mean_X = np.mean(X, axis=0)
    # # print("mean", mean_X)
    # # Center X by subtracting the mean of each column
    # X_centered = X - mean_X

    # # Calculate the standard deviation of each column
    # std_X = np.std(X, axis=0)
    # # print("sd:",std_X)
    # # Standardize X by dividing each centered column by its respective standard deviation
    # X_standardized = X_centered / std_X
    # for xj in X:
        # print("Norm xj:", np.linalg.norm(xj))
    # for xj in X_standardized.T:
    #     xj /= np.linalg.norm(xj)
    # for xj in X_standardized.T:
    #     print("Norm xj:", np.linalg.norm(xj))
    epsilon = np.random.normal(0, 1, m).reshape((-1,1))  # Normally distributed noise

    # Compute the target variable Y
    true_beta = true_beta.reshape((-1,1))
    
    Y = np.dot(X, true_beta) + epsilon
    return X, Y

if __name__ == "__main__":
    x, y = generate(5,3,np.array([0,0,0]))
    print(x)
    for xj in x.T:
        print("Norm xj:", np.linalg.norm(xj))
    print("______________")
    print(y)