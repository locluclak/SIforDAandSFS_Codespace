import numpy as np
from gendata import generate
from scipy.optimize import linprog
from Wasserstein import createOMEGA
from Wasserstein import convert
import ForwardSelection as FS

from mpmath import mp
mp.dps = 500

# global eta 
# global a
# global b 

def computeGamma(ns, nt, T):
    col0 = np.concatenate((np.zeros((ns, ns)), np.zeros((nt, ns))),axis = 0)
    col1 = np.concatenate((ns*T, np.identity(nt)), axis=0)
    return np.concatenate((col0, col1), axis = 1)
     
def FSinterval(X, Y_, gamma, SELECTION_F, aa, bb, eta):
    n_sample, n_fea = X.shape
    A=[]
    b_=[]

    Y = np.dot(gamma, Y_)

    
    k = len(SELECTION_F)

    for step in range(1,k+1):
        rss_row = []     
        # s1
        #compute K1

        I = np.identity(n_sample)
        Xfs = X[:, sorted(SELECTION_F[:step])].copy()

        K1 = I - np.dot(
                    np.dot(
                        Xfs, np.linalg.inv(np.dot(Xfs.T, Xfs))
                        ), Xfs.T)
        s1 = np.sign(np.dot(K1, Y)).copy()
        rss_row.append(np.dot(s1.T, K1)[0].copy())

        if step == 1:
            for i in range(len(s1)):
                e_ = np.zeros((len(s1),1))
                e_[i][0] = s1[i][0].copy()
                A.append(-1*np.dot(e_.T, K1)[0])
                b_.append(0)
        # rss_row_=0
    
        A.append(-1*np.dot(s1.T, K1)[0])
        b_.append(0)

        for otherfea in range(n_fea):
            if otherfea not in SELECTION_F[:step]:
                Xtemp = X[:, sorted(SELECTION_F[:step - 1] +[otherfea])].copy()
                I = np.identity(n_sample)
                K = I - np.dot(
                            np.dot(
                                Xtemp, np.linalg.inv(np.dot(Xtemp.T, Xtemp))
                                ), Xtemp.T)
                s = np.sign(np.dot(K, Y)).copy()
                rss_row.append(np.dot(s.T, K)[0].copy())

                A.append(-1*np.dot(s.T, K)[0])
                b_.append(0)

                if step == 1:
                   
                    for i in range(len(s)):
                        e_ = np.zeros((len(s),1))
                        e_[i][0] = s[i][0]
                        A.append(-1*np.dot(e_.T, K)[0])
                        b_.append(0)
            
        rss_row = np.array(rss_row)

        for RSSf in range(1, len(rss_row)):

            A.append((rss_row[0] - rss_row[RSSf]).copy())
            b_.append(0)

    
    A = np.array(A)
    b_ = np.array(b_).reshape((-1,1))
    # print(A.shape)

    #test element
    # itest = np.random.choice(SELECTION_F)
    # e = np.zeros((n_fea, 1))
    # e[itest][0] = 1

    # eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))  
    # eta = eta.reshape((-1,1))
    
    # etaTy = np.dot(eta.T, Y).item()

    Sigma = np.identity(n_sample)

    etaT_Sigma_eta = np.dot(eta.T , np.dot(Sigma, eta)).item()
    
    # c = np.dot(Sigma, eta) / etaT_Sigma_eta
    # I = np.identity(n_sample)
    # z = np.dot((I - np.dot(c,eta.T)), Y)

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

def run(iterr = 0):
    true_beta1 = np.array([0,0,0,0])
    true_beta2 = np.array([0,0,0,0])

    ns = 30
    nt = 7

    p = len(true_beta1)

    Xs, Ys = generate(ns, p, true_beta=true_beta1)
    Xt, Yt = generate(nt, p, true_beta=true_beta2)

    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    X = np.concatenate((Xs, Xt), axis= 0)
    YsYt = np.concatenate((Ys, Yt), axis= 0)

    Sigma_ = np.identity(ns+nt) # sigma of YsYt

    OMEGA = createOMEGA(ns,nt).copy()

    #SignYsYt
    Sign_obs = 0

    #cost vector
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
            cost_ += Sign * OMEGA @ XsXt_[:, [i]]

    # Sign_obs = np.sign(np.dot(OMEGA , XsXt_[:, [i]]))
    # theta = Sign_obs * OMEGA
    # cost = theta @ YsYt

   

    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 

    S = convert(ns,nt)

    #remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()

    #bound_x = [(0, None) for _ in range(n*m)]

    res = linprog(cost, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                        A_eq = S_, b_eq = h_, method = 'simplex', 
                        options={'maxiter': 10000})
    #Transport Map

    Tobs = res.x.reshape((ns,nt))
    gamma = computeGamma(ns, nt, Tobs)

    # Mobs
    basis_var = res.basis

    # Mcobs
    nonbasis_var = np.delete(np.array(range(ns*nt)), basis_var)

    theta_Mobs = theta[basis_var,:]

    t_Mobs = np.dot(np.linalg.inv(S_[:, basis_var]), h_)

    Xtilde = np.dot(gamma, X)
    SELECTION_F,r = FS.fixedSelection(np.dot(gamma, YsYt), Xtilde, 2)
    # eta = np.dot(t_Mobs.T, theta_Mobs).reshape((-1,1))
    #test element
    jtest = np.random.choice(SELECTION_F)
    e = np.zeros((p, 1))
    e[jtest][0] = 1

    XtildeA= Xtilde.copy()#[:,SELECTION_F].copy()
    eta = np.dot(e.T , np.dot(np.linalg.inv(np.dot(XtildeA.T, XtildeA)), XtildeA.T)) 
    eta = eta.reshape((-1,1))
  

    I_nplusm = np.identity(ns+nt)

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

    u = cost_ + np.dot(theta, a)
    v = np.dot(theta, b)

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

    Vminus3, Vplus3 = FSinterval(XtildeA, YsYt, gamma, SELECTION_F, a, b, eta)
    Vminus = max(Vminus, Vminus3)
    Vplus = min(Vplus, Vplus3)

    etaT_YsYt = np.dot(eta.T, YsYt).item()
    # print("etaY",etaT_YsYt)

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