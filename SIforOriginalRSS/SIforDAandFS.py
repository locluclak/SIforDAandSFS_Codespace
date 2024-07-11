import numpy as np
from gendata import generate
from scipy.optimize import linprog
import ForwardSelection as FS

from mpmath import mp
mp.dps = 500

def FSinterval(X, Y_, gamma, SELECTION_F, aa, bb, eta):
    n_sample, n_fea = X.shape
    A=[]
    b_=[]

    Y = np.dot(gamma, Y_)

    k = len(SELECTION_F)

    I = np.identity(n_sample)   

    for step in range(1, k+1):
        I = np.identity(n_sample)
        X_Mk_1 = X[:, sorted(SELECTION_F[:step - 1])].copy()
        P_pp_Mk_1 = I - np.dot(np.dot(X_Mk_1, np.linalg.inv(np.dot(X_Mk_1.T, X_Mk_1))), X_Mk_1.T) 

        Xjk = X[:, [SELECTION_F[step-1]]].copy()
        sign_projk = np.sign(np.dot(Xjk.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
        
        projk = sign_projk*(np.dot(Xjk.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xjk))
    
        if step == 1:
            A.append(-1*projk[0].copy())
            b_.append(0)
        for otherfea in range(n_fea):
            if otherfea not in SELECTION_F[:step]:

                Xj = X[:, [otherfea]].copy()
                sign_proj = np.sign(np.dot(Xj.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
                proj = sign_proj*(np.dot(Xj.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xj))
                # print(f"__{otherfea}: Proj: {proj.dot(Y).item()} RSS: {np.linalg.norm(P_pp_Mj.dot(Y))**2}")
                # if step == 1:
                #     A.append(-1*proj[0].copy())
                #     b_.append(0)
                A.append(-1*(projk-proj)[0].copy())
                b_.append(0)
                A.append(-1*(projk+proj)[0].copy())
                b_.append(0)
    A = np.array(A)
    b_ = np.array(b_).reshape((-1,1))

    # Deviation of bunch of Y after transforming 
    # Sigma = gamma.dot(np.identity(n_sample).dot(gamma.T))
    Sigma = np.identity(n_sample) 

    etaT_Sigma_eta = np.dot(eta.T , np.dot(Sigma, eta)).item()

    Ac = np.dot(A, np.dot(gamma, bb))
    Az = np.dot(A, np.dot(gamma, aa))

    Vminus = np.NINF
    Vplus = np.inf

    for j in range(len(b_)):
        left = Ac[j][0]
        right = b_[j][0] - Az[j][0]

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

def run(num_samples, iterr = 0):
    true_beta = np.array([0, 0, 0, 0])
    # print(num_samples)
    # number of sample

    n = num_samples
    p = len(true_beta) # number of features

    # Generate data
    X, Y = generate(n, p, true_beta=true_beta)
    Sigma_ = np.identity(n)

    # Select best feature of model
    SELECTION_F,r = FS.fixedSelection(Y, X, 2)

    # X_M = Xtilde[:, sorted([x for x in range(p) if x not in SELECTION_F])].copy()
    X_M = X[:, sorted(SELECTION_F)].copy()

    # Compute eta
    jtest = np.random.choice(range(len(SELECTION_F)))
    e = np.zeros((len(SELECTION_F), 1))
    e[jtest][0] = 1

    eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T)) 
    # # Compute eta
    # jtest = np.random.choice(SELECTION_F)
    # e = np.zeros((p, 1))
    # e[jtest][0] = 1


    
    # eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(XtildeA.T, XtildeA)), XtildeA.T)) 
    eta = eta.reshape((-1,1))
    
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma_) , eta).item()
    
    #Identity
    I_nplusm = np.identity(n)
    
    #Change of var y = a + bz
    b = np.dot(Sigma_, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

    Vminus, Vplus = FSinterval(X, Y, np.identity(n), SELECTION_F, a, b, eta)
    
    #Test statistic:
    etaT_YsYt = np.dot(eta.T, Y).item()

    # compute cdf of truncated gaussian distribution
    numerator = mp.ncdf(etaT_YsYt / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)

    return selective_p_value

if __name__ == "__main__":
    for i in range(10):
        print(run(100,0)) 