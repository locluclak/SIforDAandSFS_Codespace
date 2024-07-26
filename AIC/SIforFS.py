import numpy as np
from gendata import generate
from scipy.optimize import linprog
import ForwardSelection as FS
import intersection
from time import time
from mpmath import mp
mp.dps = 500


def list_residualvec(X, Y) -> list:
    # Create 1 ... p matrixes which multiplies Y to get "best k feature residual vector"
    lst_Portho = []
    lst_SELEC_k = []
    n = Y.shape[0]
    for k in range(0, X.shape[1] + 1):
        selec_k = FS.fixedSelection(Y, X, k)[0]
        lst_SELEC_k.append(selec_k)
        X_Mk = X[:, sorted(selec_k)].copy()
        lst_Portho.append(np.identity(n) - np.dot(np.dot(X_Mk, np.linalg.pinv(np.dot(X_Mk.T, X_Mk))), X_Mk.T))
    return lst_SELEC_k, lst_Portho

def FSinterval(X, Y, K, lst_SELEC_k, lst_Portho, aa, bb):
    n_sample, n_fea = X.shape

    A=[]
    b=[]

    I = np.identity(n_sample)   

    for step in range(1, K+1):
        P_pp_Mk_1 = lst_Portho[step - 1]
        Xjk = X[:, [lst_SELEC_k[step][-1]]].copy()
        sign_projk = np.sign(np.dot(Xjk.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
        
        projk = sign_projk*(np.dot(Xjk.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xjk))
        # print("Norm:", np.linalg.norm(P_pp_Mk_1.dot(Xjk)))
        if step == 1:
            A.append(-1*projk[0].copy())
            b.append(0)
        for otherfea in range(n_fea):
            if otherfea not in lst_SELEC_k[step]:

                Xj = X[:, [otherfea]].copy()
                sign_proj = np.sign(np.dot(Xj.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
                proj = sign_proj*(np.dot(Xj.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xj))

                A.append(-1*(projk-proj)[0].copy())
                b.append(0)
                A.append(-1*(projk+proj)[0].copy())
                b.append(0)


    A = np.array(A)
    b = np.array(b).reshape((-1,1))

    Ac = np.dot(A,  bb)
    Az = np.dot(A,  aa)

    Vminus = np.NINF
    Vplus = np.inf

    for j in range(len(b)):
        left = Ac[j][0]
        right = b[j][0] - Az[j][0]

        if abs(right) < 1e-14:
            right = 0
        if abs(left) < 1e-14:
            left = 0
        
        if left == 0:
            if right < 0:
                print('Error')
        else:
            temp = right / left
            if left > 0: 
                Vplus = min(Vplus, temp)
            else:
                Vminus = max(Vminus, temp)
    return Vminus, Vplus
def AICinterval(X, Y, Portho, K, aa, bb,etay):
    n_sample, n_fea = X.shape

    O_bset = Portho[K].T.dot(Portho[K])
    sigma2 = 1 / n_sample * np.sum((Y - np.mean(Y))**2)
    A = []

    for step in range(1, n_fea + 1):
        if step != K:
            O = Portho[step].T.dot(Portho[step])

            a = 1/sigma2 * (bb.T.dot(O_bset.dot(bb)) - bb.T.dot(O.dot(bb)) ).item()

            b = 1/sigma2 * (bb.T.dot(O_bset.dot(aa)) + aa.T.dot(O_bset.dot(bb))
                           - bb.T.dot(O.dot(aa))      - aa.T.dot(O.dot(bb)) ).item()

            c = 1/sigma2 * (aa.T.dot(O_bset.dot(aa)) - aa.T.dot(O.dot(aa)) ).item() + 2*float(K - step)
            A.append(np.array([a, b, c].copy()))
    A = np.array(A)

    intervals = [] 
    for i in A:
        a, b, c = i
        intervals.append(intersection.solvequadra(a,b,c))
    # print(A)
    # print("each:",intervals)
    intervals = intersection.Intersection(intervals)
    # print("each:",intervals)
    
    return intervals 
def run(num_samples, iterr = 0):
    try:
        # tstart = time()
        # seed = 1847649695
        seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(seed)
        # print("Seed:",seed)
        true_beta = np.array([0,0,0,0])

        # number of sample
        n = num_samples
        p = len(true_beta) # number of features

        # Generate data
        X, Y = generate(n, p, true_beta=true_beta)
        
        # Covariance matrix of Y
        Sigma_ = np.identity(n)*(np.sum(true_beta**2) + 1)
        # print("Generating time: ", time() - tstart)
        # tstart = time()
        # Select best feature of model
        lst_SELECk, lst_P = list_residualvec(X, Y)
        # print("Finding all k steps time: ", time() - tstart)
        
        # tstart = time()
        SELECTION_F = FS.SelectionAIC(Y, X)
        # X_M = Xtilde[:, sorted([x for x in range(p) if x not in SELECTION_F])].copy()
        X_M = X[:, sorted(SELECTION_F)].copy()

        # Compute eta
        jtest = np.random.choice(range(len(SELECTION_F)))
        e = np.zeros((len(SELECTION_F), 1))
        e[jtest][0] = 1

        eta = np.dot(e.T , np.dot(np.linalg.pinv(np.dot(X_M.T, X_M)), X_M.T)) 
        eta = eta.reshape((-1,1))
        
        etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma_) , eta).item()
        
        #Identity
        I_nplusm = np.identity(n)
        
        #Change of var y = a + bz
        b = np.dot(Sigma_, eta) / etaT_Sigma_eta
        a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

        #Test statistic:
        etaT_Y = np.dot(eta.T, Y).item()
        # print("Test statistic time: ", time() - tstart)

        # tstart = time()
        Vminus, Vplus = FSinterval(X, Y, len(SELECTION_F), lst_SELECk, lst_P, a, b)
        # print("Linear interval time: ", time() - tstart)
        # tstart = time()
        quadratic_interval = AICinterval(X, Y, lst_P, len(SELECTION_F), a, b,etaT_Y)
        # print("Quadratic time: ", time() - tstart)

        # print(f"etay: {etaT_Y}")
        # print("Quadratic interval",quadratic_interval)
        # print("Intervals k:",Vminus, Vplus)

        # tstart = time()
        u_poly = intersection.Intersec_quad_linear(quadratic_interval, ((Vminus, Vplus),))
        # Arena of truncate PDF
        denominator = 0
        numerator = None

        # for poly in u_poly:
        #   print(f"({np.round(poly[0], 5)}, {np.round(poly[1], 5)})", end=', ')
        # print()
        # print("Final interval:",u_poly)
        for poly in u_poly:
            leftside, rightside = poly
            if leftside <= etaT_Y <= rightside:
                numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
            denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))



        cdf = float(numerator / denominator)

        # compute two-sided selective p_value
        selective_p_value = 2 * min(cdf, 1 - cdf)
        # print("pvalue time:", time() -tstart)
        return selective_p_value

        
    except:
        print("\nSeed error:",seed,"\n")
if __name__ == "__main__":
    for i in range(1):
        print(run(1000,0)) 