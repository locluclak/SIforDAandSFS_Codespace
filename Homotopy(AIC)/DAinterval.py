import Wasser
import numpy as np
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