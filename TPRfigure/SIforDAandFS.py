import numpy as np
from gendata import generate
from scipy.optimize import linprog
from Wasser import createOMEGA
from Wasser import convert
from Wasser import computeGamma
import ForwardSelection as FS

from mpmath import mp
mp.dps = 500

def interval_Wasser(ns, nt, XsXt_, res, theta, Sign_obs, cost_, S_, h_, aa, bb):
    OMEGA = createOMEGA(ns,nt).copy()

    basis_var = res.basis
    nonbasis_var = np.delete(np.array(range(ns*nt)), basis_var)

    t_Mobs = np.dot(np.linalg.inv(S_[:, basis_var]), h_)
    theta_Mobs = theta[basis_var,:]
    
    v1 = Sign_obs * np.dot(OMEGA, aa)
    v2 = Sign_obs * np.dot(OMEGA, bb)

    Vminus = np.NINF
    Vplus = np.Inf
    for j in range(v1.size):
        right = -v1[j][0]
        left = v2[j][0]
        
        if abs(right) < 1e-14:
            right = 0
        if abs(left) < 1e-14:
            left = 0
        
        if left == 0:
            if right > 0:
                print("Error")
        else:
            temp = right / left
            if left > 0:
                Vminus = max(temp, Vminus)
            else:
                Vplus = min(temp, Vplus)

    u = cost_ + np.dot(theta, aa)
    v = np.dot(theta, bb)

    u_ = (u[nonbasis_var,:].T - np.dot(u[basis_var,:].T , np.dot(np.linalg.inv(S_[:, basis_var]) , S_[:, nonbasis_var]))).T
    v_ = (v[nonbasis_var,:].T - np.dot(v[basis_var,:].T , np.dot(np.linalg.inv(S_[:, basis_var]) , S_[:, nonbasis_var]))).T


    for j in range(u_.size):
        right = -u_[j][0]
        left = v_[j][0]
        if abs(right) < 1e-14:
            right = 0
        if abs(left) < 1e-14:
            left = 0
        if left == 0:
            if right > 0:
                print("Error")
        else:
            temp = right / left
            if left > 0:
                Vminus = max(temp, Vminus)            
            else:
                Vplus = min(temp, Vplus)    
    return Vminus, Vplus

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
    true_beta1 = np.array([0.5, 0, 0]) #source's beta
    true_beta2 = np.array([0.5, 0, 0]) #target's beta
    # print(num_samples)
    # number of sample
    ns = int(num_samples * 0.8) # source ~ 80%
    nt = num_samples - ns # target ~ 20%

    p = len(true_beta1) # number of features

    # Generate data
    Xs, Ys = generate(ns, p, true_beta=true_beta1)
    Xt, Yt = generate(nt, p, true_beta=true_beta2)

    #Concatenate data (X, Y)
    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    #Concatenate data into a bunch (XsYs, XtYt).T
    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    #Bunch of Xs and Xt
    X = np.concatenate((Xs, Xt), axis= 0)
    # Bunch of Ys & Yt
    YsYt = np.concatenate((Ys, Yt), axis= 0)

    # Deviation of Y
    Sigma_ = np.identity(ns+nt) 

    OMEGA = createOMEGA(ns,nt).copy()

    #SignYsYt
    Sign_obs = 0

    #Cost vector
    cost = 0
    cost_ = 0
    for i in range(p+1):
        Sign = np.sign(np.dot(OMEGA , XsXt_[:, [i]]))
        cost += Sign * np.dot(OMEGA , XsXt_[:, [i]])
        #theta YsYt and c''
        if i == p:
            Sign_obs = Sign
            theta = Sign_obs * OMEGA
        else:
            #cost''
            cost_ += Sign * OMEGA.dot(XsXt_[:, [i]])
    
    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = convert(ns,nt)
    
    #remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()

    # Solve wasserstein distance
    res = linprog(cost, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                        A_eq = S_, b_eq = h_, method = 'simplex', 
                        options={'maxiter': 10000})
    # Transport Map
    Tobs = res.x.reshape((ns,nt))
    gamma = computeGamma(ns, nt, Tobs)
    
    # Bunch of Xs Xt after transforming
    Xtilde = np.dot(gamma, X)

    # Select best feature of model
    SELECTION_F,r = FS.fixedSelection(np.dot(gamma, YsYt), Xtilde, 2)

    # X_M = Xtilde[:, sorted([x for x in range(p) if x not in SELECTION_F])].copy()
    X_M = Xtilde[:, sorted(SELECTION_F)].copy()

    # Compute eta
    jtest = np.random.choice(range(len(SELECTION_F)))
    e = np.zeros((len(SELECTION_F), 1))
    e[jtest][0] = 1

    eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T)) 

    XtildeA= Xtilde.copy()
    
    # eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(XtildeA.T, XtildeA)), XtildeA.T)) 
    eta = eta.reshape((-1,1))
    
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma_) , eta).item()
    
    #Identity
    I_nplusm = np.identity(ns+nt)
    
    #Change of var y = a + bz
    b = np.dot(Sigma_, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), YsYt)

    # Interval of z1, z2
    Vminus12, Vplus12 = interval_Wasser(ns, nt, XsXt_, res, theta, Sign_obs, cost_, S_, h_, a, b)
    # Interval of z3
    Vminus3, Vplus3 = FSinterval(XtildeA, YsYt, gamma, SELECTION_F, a, b, eta)
    
    Vminus = max(Vminus12, Vminus3)
    Vplus = min(Vplus12, Vplus3)
    # print(f"z12 [{Vminus12}, {Vplus12}]")
    # print(f"z3 [{Vminus3}, {Vplus3}]")
    #Test statistic:
    etaT_YsYt = np.dot(eta.T, YsYt).item()

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