import numpy as np
from scipy.optimize import linprog
from gendata import generate
import Wasser
import ForwardSelection as FS

from mpmath import mp
import SFSinterval
import DAinterval
mp.dps = 500

import time

def DA_Wasser(ns, nt, S_, h_, X_):
    OMEGA = Wasser.createOMEGA(ns,nt).copy()

    #Cost vector
    cost = 0

    p = X_.shape[1] - 1 
    for i in range(p+1):
        cost += abs(np.dot(OMEGA , X_[:, [i]]))
        
    # Solve wasserstein distance
    res = linprog(cost, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                        A_eq = S_, b_eq = h_, method = 'simplex', 
                        options={'maxiter': 10000})
    # Transport Map
    Tobs = res.x.reshape((ns,nt))
    gamma = Wasser.computeGamma(ns, nt, Tobs)
    return {"gamma": gamma, "basis": res.basis}

def run(num_samples = 30):
    # seed = 3113993449
    seed = int(np.random.rand() * (2**32 - 1))
    np.random.seed(seed)
    # print("Seed:",seed)

    true_betaS = np.array([0, 0, 0 ]) #source's beta
    true_betaT = np.array([0, 0, 0 ]) #target's beta

    # number of sample
    # ns = int(num_samples * 0.8) # source ~ 80%
    # nt = num_samples - ns       # target ~ 20%
    nt = 10               # target = 10
    ns = num_samples - nt # source = n_sample - target

    p = len(true_betaS) # number of features

    # Generate data
    Xs, Ys = generate(ns, p, true_beta=true_betaS)
    Xt, Yt = generate(nt, p, true_beta=true_betaT)

    #Concatenate data (X, Y)
    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    #Concatenate data into a bunch (XsYs, XtYt).T
    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    #Bunch of Xs and Xt
    X = np.concatenate((Xs, Xt), axis= 0)
    # Bunch of Ys & Yt
    Y = np.concatenate((Ys, Yt), axis= 0)

    # Deviation of Y
    Sigma = np.diag([np.sum(true_betaS**2) + 1]*ns + [np.sum(true_betaT**2) + 1]*nt)

    # Linear program part
    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = Wasser.convert(ns,nt)
    # remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()
    # Gamma drives source data to target data 
    GAMMA,res_linprog = DA_Wasser(ns, nt, S_, h_, XsXt_).values()

    # Bunch of Xs Xt after transforming
    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)

    # Best model 
    SELECTION_F = FS.fixedSelection(Ytilde, Xtilde, 2)[0]
    lst_SELECk, lst_P = SFSinterval.list_residualvec(Xtilde, Ytilde)


    X_M = Xt[:, sorted(SELECTION_F)].copy()

    # Compute eta
    jtest = np.random.choice(range(len(SELECTION_F)))
    e = np.zeros((len(SELECTION_F), 1))
    e[jtest][0] = 1

    # Zeta cut off source data in Y
    Zeta = np.concatenate((np.zeros((nt, ns)), np.identity(nt)), axis = 1)
    
    # eta constructed on Target data
    eta = np.dot(e.T , np.dot(np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T), Zeta)) 
    eta = eta.reshape((-1,1))
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma) , eta).item()
    
    # Change y = a + bz
    I_nplusm = np.identity(ns+nt)
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

    # Test statistic
    etaT_Y = np.dot(eta.T, Y).item()

    # Interval of z1, z2 DA
    Vminus12, Vplus12 = DAinterval.interval_DA(ns, nt, XsXt_, res_linprog, S_, h_, a, b)
    # Interval of z3 SFS
    Vminus3, Vplus3 = SFSinterval.interval_SFS(Xtilde, Ytilde, 
                                    len(SELECTION_F),
                                    lst_SELECk, lst_P,
                                    GAMMA.dot(a), GAMMA.dot(b))
    
    Vminus = max(Vminus12, Vminus3)
    Vplus = min(Vplus12, Vplus3)
    # print(f"({Vm_}, {Vp_})")


    # compute cdf of truncated gaussian distribution
    numerator = mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)

    # print(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T), Zeta), Y))
    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)
    return selective_p_value
if __name__ == "__main__":
    for i in range(1):
        st = time.time()
        print(run(150))
        en = time.time()
        print(f"Time step {i}: {en - st}")
