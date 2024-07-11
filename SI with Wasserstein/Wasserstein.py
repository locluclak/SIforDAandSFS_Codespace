import numpy as np
from gendata import generate
from scipy.optimize import linprog


from mpmath import mp
mp.dps = 500

def createOMEGA(n, m):
    vec1m = np.ones((m,1))
    vec0m = np.zeros((m,1))

    Im = -1*np.identity(m)

    vecIm = Im
    for i in range(n-1):
        vecIm = np.append(vecIm, Im, axis = 0)

    iden_vec1m = None
    for i in range(n):
        temp = None
        for j in range(n):
            if i == j:
                if temp is None:
                    temp = vec1m
                else:
                    temp = np.hstack((temp,vec1m))
            else:
                if temp is None:
                    temp = vec0m
                else:
                    temp = np.hstack((temp,vec0m))

        if iden_vec1m is None:
            iden_vec1m = temp
        else:
            iden_vec1m = np.vstack((iden_vec1m, temp))
    return np.hstack((iden_vec1m, vecIm))

def convert(m, n):
    A = np.arange(m*n)
    A = A.reshape((m,n))
    B = []
    
    for row in A:
        temp = np.zeros(m*n)
        for ele in row:
            temp[ele] = 1
        B.append(temp)
    for col in A.T:
        temp = np.zeros(m*n)
        for ele in col:
            temp[ele] = 1
        B.append(temp)
    return np.array(B)

def run(step = 0):
    true_beta1 = np.array([])
    true_beta2 = np.array([])

    n = 30
    m = 7

    k = len(true_beta1)

    Xs, Ys = generate(n, k, true_beta=true_beta1)
    Xt, Yt = generate(m, k, true_beta=true_beta2)

    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    YsYt = XsXt_[:, [k]]

    Sigma_ = np.identity(n+m)

    OMEGA = createOMEGA(n,m).copy()

    #SignYsYt
    Sign_obs = 0

    #cost vector
    cost = 0
    cost_ = 0
    for i in range(k+1):
        Sign = np.sign(np.dot(OMEGA , XsXt_[:, [i]]))
        cost += Sign * np.dot(OMEGA , XsXt_[:, [i]])
        #theta YsYt and c''
        if i == k:
            Sign_obs = Sign
            theta = Sign_obs * OMEGA
        else:
            #cost''
            cost_ +=  np.dot(Sign * OMEGA , XsXt_[:, [i]])

    # Sign_obs = np.sign(np.dot(OMEGA , XsXt_[:, [i]]))
    # theta = Sign_obs * OMEGA
    # cost = theta @ YsYt

   

    h = np.concatenate((np.ones((n, 1))/n, np.ones((m,1))/m), axis = 0) 

    S = convert(n,m)

    #remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()

    #bound_x = [(0, None) for _ in range(n*m)]

    res = linprog(cost, A_eq = S_, b_eq = h_, method = 'simplex', options={'maxiter': 10000})

    # Mobs
    basis_var = res.basis

    # Mcobs
    nonbasis_var = np.delete(np.array(range(n*m)), basis_var)

    theta_Mobs = theta[basis_var,:]

    t_Mobs = np.dot(np.linalg.inv(S_[:, basis_var]), h_)

    eta = np.dot(t_Mobs.T, theta_Mobs).reshape((-1,1))

    # W = optimize.linprog

    # W(Pn, Qn) = t_Mobs.T @ c'' + eta.T @ YsYt
    # np.dot(t_Mobs.T , cost_[basis_var,:]) + np.dot(eta.T, YsYt)

    I_nplusm = np.identity(n+m)

    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma_) , eta).item()

    # a + bz
    # z + c(eta.T @ Y)
    # a = q(Ys, Yt)
    b = np.dot(Sigma_, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), YsYt)

    v1 = Sign_obs * np.dot(OMEGA, a)
    v2 = Sign_obs * np.dot(OMEGA, b)

    Vminus = np.NINF
    Vplus = np.Inf
    for j in range(v1.size):
        right = -v1[j][0]
        left = v2[j][0]
        
        # if abs(right) < 1e-14:
        #     right = 0
        # if abs(left) < 1e-14:
        #     left = 0
        
        if left == 0:
            if right > 0:
                print("Error")
        else:
            temp = right / left
            if left > 0:
                Vminus = max(temp, Vminus)
            else:
                Vplus = min(temp, Vplus)

    u = np.dot(theta, a)
    v = np.dot(theta, b)

    u_ = (u[nonbasis_var,:].T - np.dot(u[basis_var,:].T , np.dot(np.linalg.inv(S_[:, basis_var]) , S_[:, nonbasis_var]))).T
    v_ = (v[nonbasis_var,:].T - np.dot(v[basis_var,:].T , np.dot(np.linalg.inv(S_[:, basis_var]) , S_[:, nonbasis_var]))).T

    # Vminus = np.NINF
    # Vplus = np.Inf
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
        print(run())    