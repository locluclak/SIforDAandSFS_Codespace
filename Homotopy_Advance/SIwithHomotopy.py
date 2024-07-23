import numpy as np
from gendata import generate
from scipy.optimize import linprog
import Wasser
import ForwardSelection as FS
import time

from mpmath import mp
mp.dps = 500


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
    # return gamma 

def interval_DA(ns, nt, X_, res, S_, h_, aa, bb):
    OMEGA = Wasser.createOMEGA(ns,nt).copy()

    #SignYsYt
    Sign_obs = 0

    #Cost vector
    cost = 0
    cost_ = 0

    p = X_.shape[1] - 1 
    for i in range(p+1):
        Sign = np.sign(np.dot(OMEGA , X_[:, [i]]))
        cost += Sign * np.dot(OMEGA , X_[:, [i]])
        #theta YsYt and c''
        if i == p:
            Sign_obs = Sign
            theta = Sign_obs * OMEGA
        else:
            #cost''
            cost_ += Sign * OMEGA.dot(X_[:, [i]])
    

    basis_var = res
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


def interval_SFS(X_nontrans, Y_nontrans, gamma, SELECTION_F, aa, bb):
    # Xtilde Ytilde
    X = gamma.dot(X_nontrans)
    Y = gamma.dot(Y_nontrans)

    n_sample, n_fea = X.shape
    A=[]
    b=[]

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
            b.append(0)
        for otherfea in range(n_fea):
            if otherfea not in SELECTION_F[:step]:

                Xj = X[:, [otherfea]].copy()
                sign_proj = np.sign(np.dot(Xj.T , np.dot(P_pp_Mk_1, Y)).item()).copy()
                proj = sign_proj*(np.dot(Xj.T, P_pp_Mk_1)) / np.linalg.norm(P_pp_Mk_1.dot(Xj))
                # print(f"__{otherfea}: Proj: {proj.dot(Y).item()} RSS: {np.linalg.norm(P_pp_Mj.dot(Y))**2}")
                if step == 1:
                    A.append(-1*proj[0].copy())
                    b.append(0)
                A.append(-1*(projk-proj)[0].copy())
                b.append(0)
                A.append(-1*(projk+proj)[0].copy())
                b.append(0)
    A = np.array(A)
    b = np.array(b).reshape((-1,1))

    Ac = np.dot(A, np.dot(gamma, bb))
    Az = np.dot(A, np.dot(gamma, aa))

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

def unionPoly(listofpoly, Vm, Vp):
    ls = []
    pushed = False
    for poly in listofpoly:
        Vminus_, Vplus_ = poly
        if pushed == False:
            if Vp <= Vplus_:
                pushed = True
            if Vm <= Vplus_ and Vp > Vplus_:
                ls.append((Vminus_, Vp))
                pushed = True
                continue
        ls.append(poly)

    if pushed == False:
        ls.append((Vm, Vp))
    return ls

def run(num_samples, iter = 0):
    seed = 3562088209
    # seed = int(np.random.rand() * (2**32 - 1))
    np.random.seed(seed)
    print(seed)
    true_beta1 = np.array([0, 0, 0]) #source's beta
    true_beta2 = np.array([0, 0, 0]) #target's beta
    # print(num_samples)
    # number of sample
    ns = int(num_samples * 0.8) # source ~ 80%
    nt = num_samples - ns # target ~ 20%
    # nt = 10# target = 10
    # ns = num_samples - nt # source ~ 80%

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
    Y = np.concatenate((Ys, Yt), axis= 0)

    # Deviation of Y
    Sigma_ = np.identity(ns+nt) 


    # Linear program
    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = Wasser.convert(ns,nt)
        #remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()


    # Gamma drives source data to target one 
    GAMMA = DA_Wasser(ns, nt, S_, h_, XsXt_)["gamma"]

    # Bunch of Xs Xt after transforming
    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)

    # Select 2 best features of model
    SELECTION_F,r = FS.fixedSelection(Ytilde, Xtilde, 2)

    X_M = Xt[:, sorted(SELECTION_F)].copy()

    # Compute eta
    jtest = np.random.choice(range(len(SELECTION_F)))
    e = np.zeros((len(SELECTION_F), 1))
    e[jtest][0] = 1

    # Zeta cut off source data in Y
    Zeta = np.concatenate((np.zeros((nt, ns)), np.identity(nt)), axis = 1)

    eta = np.dot(e.T , np.dot(np.dot(np.linalg.inv(np.dot(X_M.T, X_M)), X_M.T), Zeta)) 
    eta = eta.reshape((-1,1))
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma_) , eta).item()
    
    #Identity
    I_nplusm = np.identity(ns+nt)
    
    #Change of var y = a + bz
    b = np.dot(Sigma_, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)
    
    # Test statistic
    etaT_Y = np.dot(eta.T, Y).item()
    # # Compute intervals
    # Vminus12, Vplus12 = interval_DA(ns, nt, XsXt_, a, b)
    # Vminus3, Vplus3 = interval_SFS(X, Y, GAMMA, SELECTION_F, a, b)
    # Vminus = max(Vminus12, Vminus3)
    # Vplus = min(Vplus12, Vplus3)
    u_poly = []
    
    # zrange = tuple(x*0.02 for x in range(-20*50, 20*50+1)) # [-20, 20, step = 0./04]
    z = -20
    # ztemp = z 
    zmax = 20
    # z = etaT_Y - 0.001
    # ztemp = z
    # zmax = etaT_Y + 0.001
    while z <= zmax:
        # print(z)

        z += 0.001

        if z < etaT_Y and z - 0.001 >= etaT_Y:
            z = etaT_Y
            print("Catched")
            print("Seed: ",seed)


        # print(z)
        Ydeltaz = a + b*z
        # if abs(z) >= 10:
        #     z+= 0.5
        # else:
        # print(z)

        
        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis= 1).copy()
        GAMMAdeltaz, res_linprog = DA_Wasser(ns, nt, S_, h_, XsXt_deltaz).values()

        # Bunch of Xs Xt after transforming
        Xtildeinloop = np.dot(GAMMAdeltaz, X)
        Ytildeinloop = np.dot(GAMMAdeltaz, Y)

        # Select 2 best features of model
        SELECTION_Finloop = FS.fixedSelection(Ytildeinloop, Xtildeinloop, 2)[0]
        
        # Interval of z1, z2
        Vminus12, Vplus12 = interval_DA(ns, nt, XsXt_deltaz, res_linprog, S_, h_, a, b)
        # Interval of z3
        Vminus3, Vplus3 = interval_SFS(X, Ydeltaz, GAMMAdeltaz, SELECTION_F, a, b)
        

        Vm_ = max(Vminus12, Vminus3)
        Vp_ = min(Vplus12, Vplus3)
        if Vm_ > Vp_:
            # z = ztemp
            continue
        
        if z < Vp_:
            z = Vp_
            # ztemp = z
            # print(f"Finded: {z}")
        # else:
        #     z = ztemp

        if SELECTION_F != SELECTION_Finloop:
            continue

        u_poly = unionPoly(u_poly, Vm_, Vp_)
        # print(z)
        
    
    
    
    # Arena of truncate PDF
    denominator = 0
    numerator = None
    # print(f"etay: {etaT_Y}")
    # for poly in u_poly:
    #   print(f"({np.round(poly[0], 5)}, {np.round(poly[1], 5)})", end=', ')
    # print()
    for poly in u_poly:
        leftside, rightside = poly
        if leftside <= etaT_Y <= rightside:
            numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))


    # if numerator == None:
    #     print("Seed: ",seed)
    #     numerator = 0
    cdf = float(numerator / denominator)


    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)
    return selective_p_value
if __name__ == "__main__":
    for i in range(1):
        st = time.time()
        print(run(20, 0))
        en = time.time()
        print(f"Time step {i}: {en - st}")