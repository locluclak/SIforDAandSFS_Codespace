import numpy as np
import intersection
from mpmath import mp
mp.dps = 500


def run(iter = 0):
    # seed = 2629361877
    try:
        seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(seed)

        X = np.random.normal(0,1,3).reshape((-1, 1))
        # print("X: ", X)
        Sigma = np.identity(3)
        A = [np.zeros((3,3)) for i in range(2)]

        maxx= 0
        j=0
        for i in range(3):
            if (X[i][0]**2) > maxx:
                maxx = X[i][0]**2
                j = i

        eta = np.zeros((3,1))
        eta[j] = 1
        etaT_Sigma_eta = eta.T.dot(Sigma.dot(eta)).item()
        bb = Sigma.dot(eta) / etaT_Sigma_eta 
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
                rows.append(-1 * bb.T.dot(A[ath].dot(bb)).item())
                rows.append(-1 * (bb.T.dot(A[ath].dot(aa)) + aa.T.dot(A[ath].dot(bb))).item())
                rows.append(-1 * aa.T.dot(A[ath].dot(aa)).item())

                ath += 1
                Acons.append(np.array(rows).copy())
        Acons = np.array(Acons)
        intervals = []
        for i in Acons:
            a, b, c = i
            intervals.append(intersection.solvequadra(a,b,c))
        intervals = intersection.Intersection(intervals)
        
        etaT_Y = np.dot(eta.T, X).item()

        # Arena of truncate PDF
        denominator = 0
        numerator = None

        for poly in intervals:
            leftside, rightside = poly
            if leftside <= etaT_Y <= rightside:
                numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
            denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))


        cdf = float(numerator / denominator)


        # compute two-sided selective p_value
        selective_p_value = 2 * min(cdf, 1 - cdf)
        return selective_p_value
    except:
        raise ValueError(seed)

if __name__ == "__main__":
    print(run())