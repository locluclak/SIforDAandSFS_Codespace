import numpy as np
from sklearn.linear_model import LinearRegression

def Selection(Y, X):
    #Y = Y.flatten()
    Cp = np.inf
    n = X.shape[0]

    fullmodel = np.linalg.inv(X.T @ X) @ X.T @ Y

    sigma2 = RSS(Y,X, fullmodel) / X.shape[1]

    for i in range(0, X.shape[1] + 1):
        sset, rss = fixedSelection(Y, X, i)
        d = i
        cp = (rss + 2*d*sigma2) / n
        if cp < Cp:
            bset = sset
            Cp = cp
    return bset
def SelectionAIC(Y,X):
    AIC = np.inf
    n = X.shape[0]

    sigma2 = 1 / n * np.sum((Y - np.mean(Y))**2) 
    for i in range(1, X.shape[1] + 1):
        sset, rss = fixedSelection(Y, X, i)
        d = i
        aic = rss/sigma2 + 2*i
        # print(aic)
        if aic < AIC:
            bset = sset
            AIC = aic
    return bset
def oneSelection(Y,X):
    rest = list(range(X.shape[1]))
    rss = np.inf
    for fea in rest:
        
        #select nessesary data
        X_temp = X[:, [fea]].copy()
        #create linear model

        #calculate rss of model
        rss_temp = RSSwithoutcoef(Y, X_temp)
        # print("rss temp:", rss_temp)
        if rss > rss_temp:
            rss = rss_temp
            selection = fea
    return [selection]


def fixedSelection(Y, X, k):
    # if k == 1:
    #     return oneSelection(Y,X), 0
    selection = []
    rest = list(range(X.shape[1]))
    rss = np.linalg.norm(Y)**2
    #i = 1
    for i in range(1, k+1):
        rss = np.inf
        sele = selection.copy()
        selection.append(None)
        for feature in rest:
            if feature not in selection:
                #select nessesary data
                X_temp = X[:, sorted(sele + [feature])].copy()
                #create linear model

                #calculate rss of model
                rss_temp = RSSwithoutcoef(Y, X_temp)
                
                # choose feature having minimum rss and append to selection
                if rss > rss_temp:
                    rss = rss_temp
                    selection.pop()
                    selection.append(feature)
        # print("RSS of selected feature:", rss)
    return selection, rss
def RSS(Y, X, coef, intercept = 0):
    RSS = 0
    for i_sample in range(np.shape(X)[0]):
        Y_hat = np.dot(X[i_sample], coef).item() + intercept
        RSS += (Y[i_sample][0] - Y_hat)**2
    return RSS        

def RSSwithoutcoef(Y, X, intercept = 0):
    RSS = 0
    coef = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T) , Y)
    RSS = np.linalg.norm(Y - np.dot(X, coef))**2
    # print("RSS:", RSS)
    return RSS        