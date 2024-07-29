import numpy as np
from gendata import generate
from scipy.optimize import linprog
import Wasser
import ForwardSelection as FS
import time
import SFSinterval
import DAinterval

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
    # seed = 3562088209
    seed = int(np.random.rand() * (2**32 - 1))
    np.random.seed(seed)
    # print(seed)
    true_beta1 = np.array([0, 0, 0]) #source's beta
    true_beta2 = np.array([0, 0, 0]) #target's beta
    # print(num_samples)
    # number of sample
    # ns = int(num_samples * 0.8) # source ~ 80%
    # nt = num_samples - ns # target ~ 20%
    nt = 10               # target = 10
    ns = num_samples - nt # source = num_samples - target

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
    

        
    
    
    z =  -20
    zmax = 20
    while z < zmax:
        z += 0.001

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
        SELECTIONinloop = FS.fixedSelection(Ytildeinloop, Xtildeinloop, 2)[0]
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
        
        if Vm_ <= z <= Vp_:
            z = Vp_
        if sorted(SELECTIONinloop) != sorted(SELECTION_F):
            # print(f"    Continue {SELECTION_F} {SELECTIONinloop}", end= "  | ")
            # print(intervalinloop)
            continue
        # print(f"    Catch {SELECTION_F} {SELECTIONinloop}")

        # print("interval each z: ", intervalinloop)
        u_poly = unionPoly(u_poly, Vm_, Vp_)
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
        print(run(50, 0))
        en = time.time()
        print(f"Time step {i}: {en - st}")