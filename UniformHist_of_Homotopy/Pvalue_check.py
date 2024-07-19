
import scipy 
import numpy as np
import matplotlib.pyplot as plt

def main(st = 0):
    max_iteration = 1104
    list_p_value = []
    ssize = 34
    alpha = 0.05
    count = 0


    # Open the file in read mode
    with open('list_p_value.txt', 'r') as file:
        # Read each line in the file
        for line in file:
            # Convert the line to a float and append it to the list
            list_p_value.append(float(line.strip()))



    for i in list_p_value:
        if i <= alpha:
            count += 1

    print('False positive rate:', count / max_iteration)
    # print('True positive rate:', 1 - count / max_iteration)
    kstest_pvalue = scipy.stats.kstest(list_p_value, 'uniform').pvalue
    print('Uniform Kstest check:', kstest_pvalue)
    plt.hist(list_p_value)
    # Save the histogram
    plt.savefig('uniform_hist.png')
    plt.show()


    return kstest_pvalue
    
    

if __name__ == "__main__":

    loop = 1
    count_ = 0
    for i in range(loop):
        print(f"Loop {i+1}/{loop}")
        kstest = main()
        if kstest <= 0.05:
            count_ += 1
    # print(f"\n% <= 0.05: {count_}/{loop}")
    # iter = range(16)
    # with mpr.Pool() as pool:
    #     pool.map(main, iter)