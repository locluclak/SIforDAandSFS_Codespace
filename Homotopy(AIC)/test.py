from gendata import generate
import numpy as np
import matplotlib.pyplot as plt
import scipy 

true_beta = np.array([3])
n = 100000
p = 1
x, y = generate(n, p, true_beta)
kstest_pvalue = scipy.stats.kstest(x.reshape((1, n))[0], 'uniform').pvalue

print('Uniform Kstest check:', kstest_pvalue)

plt.hist(x)
plt.show()

