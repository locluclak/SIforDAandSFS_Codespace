import SIwithHomotopy
import multiprocessing as mpr
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import os
import time 
def main(st = 0):
    max_iteration = 500
    list_p_value = []
    ssize = 20
    alpha = 0.05
    count = 0
    #print("core available: ", mpr.cpu_count())
    iter = (ssize,) * max_iteration

    with mpr.Pool(initializer = np.random.seed) as pool:
        list_p_value = pool.map(SIwithHomotopy.run, iter)
    # for i in range(max_iteration):
    #     list_p_value.append(Wasserstein.run())
    filew = open("list_p_value.txt","a")

    for i in list_p_value:
        filew.write(str(i))
        filew.write("\n")
        if i <= alpha:
            count += 1
    filew.close()
    print('False positive rate:', count / max_iteration)
    # print('True positive rate:', 1 - count / max_iteration)
    kstest_pvalue = scipy.stats.kstest(list_p_value, 'uniform').pvalue
    print('Uniform Kstest check:', kstest_pvalue)
    plt.hist(list_p_value)
    # Save the histogram
    plt.savefig('HOMOTOPY_uniform_hist.png')
    plt.show()
    

    return kstest_pvalue
    
    

if __name__ == "__main__":
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    loop = 1
    st = time.time()
    for i in range(loop):
        print(f"Loop {i+1}/{loop}")
        kstest = main()
    print(f"Time: {time.time() - st}")
    # print(f"\n% <= 0.05: {count_}/{loop}")
    # iter = range(16)
    # with mpr.Pool() as pool:
    #     pool.map(main, iter)