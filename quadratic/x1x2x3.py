import numpy as np

from mpmath import mp
mp.dps = 500

def run():
    X = np.random.normal(0,1,3).reshape((-1, 1))
    Sigma = np.identity(3)
    A = [np.zeros((3,3)) for i in range(2)]

    maxx= 0
    j=0
    for i in range(3):
        if X[i][0]**2 > maxx:
            maxx = X[i][0]
            j = i
    eta = np.zeros((3,1))
    eta[j] = 1
    
    bb = Sigma.dot(eta) / (eta.T.dot(Sigma.dot(eta)))
    aa = (np.identity(3) - bb.dot(eta.T)).dot(X)  

    A[0][j][j] =1
    A[1][j][j] =1
    
    ath = 0
    for i in range(3):
        if i != j:
            A[ath][i][i] = -1
            ath += 1
    ath = 0
    Acons = []
    for i in range(3):
        if i != j:
            rows = []
            rows.append(bb.T.dot(A[ath].dot(bb)).item())
            rows.append((bb.T.dot(A[ath].dot(aa)) + aa.T.dot(A[ath].dot(bb))).item())
            rows.append(aa.T.dot(A[ath].dot(aa)).item())

            ath += 1
            Acons.append(np.array(rows).copy())
    Acons = np.array(Acons)
    print(Acons)
    return 0 

if __name__ == "__main__":
    print(run())