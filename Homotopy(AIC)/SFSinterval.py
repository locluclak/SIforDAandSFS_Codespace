import numpy as np
import ForwardSelection as FS
import intersection

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


def interval_SFS(X, Y, K, lst_SELEC_k, lst_Portho, aa, bb):
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


def AICinterval(X, Y, Portho, K, aa, bb):
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