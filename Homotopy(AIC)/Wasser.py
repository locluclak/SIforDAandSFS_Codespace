import numpy as np 

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

def computeGamma(ns, nt, T):
    col0 = np.concatenate((np.zeros((ns, ns)), np.zeros((nt, ns))),axis = 0)
    col1 = np.concatenate((ns*T, np.identity(nt)), axis=0)
    return np.concatenate((col0, col1), axis = 1)