# Sample size: 50
# Iter: 1200
# False positive rate: 0.056666666666666664
# Uniform Kstest check: 0.6623303683089086

# Sample size: 100
# False positive rate: 0.049166666666666664                                                                                                
# Uniform Kstest check: 0.046069760084252676                       
# Time: 937.4863836765289 s

# Sample size: 150
# Loop 1/1
# False positive rate: 0.05333333333333334
# Uniform Kstest check: 0.23692123572428647
# Time: 3207.9846329689026 s

# Sample size: 200
# Loop 1/1
# False positive rate: 0.041666666666666664
# Uniform Kstest check: 0.19846993018854064
# Time: 13952.33544254303 s

import matplotlib.pyplot as plt

# Sample data
sample_sizes = [50, 100, 150, 200]
fpr = [0.056666666666666664, 0.049166666666666664, 0.05333333333333334, 0.041666666666666664]
# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, fpr, marker='o', linestyle='-', label='fpr')

# Labeling the axes
plt.xlabel('Sample size')
plt.ylabel('FPR')
plt.ylim(0, 1.0)
plt.savefig('FPRs.png')

# Adding a legend
# plt.legend()

# Display the plot
plt.show()