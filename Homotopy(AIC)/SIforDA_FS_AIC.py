import numpy as np
from scipy.optimize import linprog
from gendata import generate
import Wasser
import ForwardSelection as FS
import intersection
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
    print("Seed:",seed)

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

    # Best model from 1...p models by AIC criterion
    SELECTION_F = FS.SelectionAIC(Ytilde, Xtilde)

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

    # print(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T), Zeta), Y))
    # Truncated distribution's intervals
    TD = []
    detectedinter = []

    # print("etay: ",etaT_Y) #-0.3027165070348666
    # print("OBS: ",SELECTION_F)
    z =  -20
    zmax = 20
    while z < zmax:
        z += 0.001

        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.001
                detectedinter = detectedinter[i:]
                # print("Jump", z)
                break
        if z > zmax:
            break
        print(z)

        Ydeltaz = a + b*z

        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis= 1).copy()
        GAMMAdeltaz, res_linprog = DA_Wasser(ns, nt, S_, h_, XsXt_deltaz).values()

        # Bunch of Xs Xt after transforming
        Xtildeinloop = np.dot(GAMMAdeltaz, X)
        Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)

        # Select best feature of model 
        SELECTIONinloop = FS.SelectionAIC(Ytildeinloop, Xtildeinloop)
        lst_SELECk, lst_P = SFSinterval.list_residualvec(Xtildeinloop, Ytildeinloop)


        # Interval of z1, z2 DA
        Vminus12, Vplus12 = DAinterval.interval_DA(ns, nt, XsXt_deltaz, res_linprog, S_, h_, a, b)
        # Interval of z3 SFS
        Vminus3, Vplus3 = SFSinterval.interval_SFS(Xtildeinloop, Ytildeinloop, 
                                        len(SELECTIONinloop),
                                        lst_SELECk, lst_P,
                                        GAMMAdeltaz.dot(a), GAMMAdeltaz.dot(b))
        
        Vm_ = max(Vminus12, Vminus3)
        Vp_ = min(Vplus12, Vplus3)
        # print(f"({Vm_}, {Vp_})")
        quadratic_interval = SFSinterval.AICinterval(Xtildeinloop, Ytildeinloop, 
                                                        lst_P, len(SELECTIONinloop), 
                                                        GAMMAdeltaz.dot(a), GAMMAdeltaz.dot(b))
        # print("Quadra intervals:",quadratic_interval)
        intervalinloop = intersection.Intersec_quad_linear(quadratic_interval, ((Vm_, Vp_),))
        
        detectedinter = intersection.Union(detectedinter, intervalinloop)

        if sorted(SELECTIONinloop) != sorted(SELECTION_F):
            # print(f"    Continue {SELECTION_F} {SELECTIONinloop}", end= "  | ")
            # print(intervalinloop)
            continue
        # print(f"    Catch {SELECTION_F} {SELECTIONinloop}")

        # print("interval each z: ", intervalinloop)
        TD = intersection.Union(TD, intervalinloop)
    # print(TD)
    # TD = u_poly


    # Arena of truncate PDF
    denominator = 0
    numerator = None

    for i in TD:
        leftside, rightside = i
        if leftside <= etaT_Y <= rightside:
            numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))

    cdf = float(numerator / denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)
    print("Seed", seed)
    return selective_p_value
if __name__ == "__main__":
    for i in range(1):
        st = time.time()
        print(run(150))
        en = time.time()
        print(f"Time step {i}: {en - st}")
