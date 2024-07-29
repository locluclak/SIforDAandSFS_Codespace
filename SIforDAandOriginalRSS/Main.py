import SIforDA_FS
import multiprocessing as mpr
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import os
import time
def main(st = 0):
    max_iteration = 40000
    list_p_value = []
    ssize = 40
    alpha = 0.05
    count = 0
    #print("core available: ", mpr.cpu_count())
    iter = (ssize,) * max_iteration

    with mpr.Pool(initializer = np.random.seed) as pool:
        list_p_value = pool.map(SIforDA_FS.run, iter)

    for i in list_p_value:
        if i <= alpha:
            count += 1

    print('False positive rate:', count / max_iteration)
    kstest_pvalue = scipy.stats.kstest(list_p_value, 'uniform').pvalue
    print('Uniform Kstest check:', kstest_pvalue)
    plt.hist(list_p_value)
    # Save the histogram
    plt.savefig('uniform_hist.png')
    plt.show()


    return kstest_pvalue
    
    

if __name__ == "__main__":
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    loop = 1

    for i in range(loop):
        st = time.time()
        print(f"Loop {i+1}/{loop}")
        kstest = main()
        print("Time:", time.time() - st)