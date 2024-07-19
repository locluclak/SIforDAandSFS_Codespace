# TPRs figure

# PS C:\Users\Asus\Documents\NCKH\SIforDAandSFS_Codespace\TPRfigure> py .\Main.py
# Loop 1/1
# Sample size: 50
# True positive rate: 0.17
# False negative rate: 0.83
# Uniform Kstest check: 1.0419231271071355e-16
# PS C:\Users\Asus\Documents\NCKH\SIforDAandSFS_Codespace\TPRfigure> py .\Main.py
# Loop 1/1
# Sample size: 100
# True positive rate: 0.094
# False negative rate: 0.906
# Uniform Kstest check: 0.0001029890545881265
# PS C:\Users\Asus\Documents\NCKH\SIforDAandSFS_Codespace\TPRfigure> py .\Main.py
# Loop 1/1
# Sample size: 100
# True positive rate: 0.094
# False negative rate: 0.906
# Uniform Kstest check: 0.0006503032918863729
# PS C:\Users\Asus\Documents\NCKH\SIforDAandSFS_Codespace\TPRfigure> py .\Main.py
# Loop 1/1
# Sample size: 150
# True positive rate: 0.066
# False negative rate: 0.9339999999999999
# Uniform Kstest check: 0.19580410426286265

# Loop 1/1   
# Sample size: 200
# True positive rate: 0.056                                                                                     
# False negative rate: 0.944
# Uniform Kstest check: 0.7616882224402296

import matplotlib.pyplot as plt

# Sample data
sample_sizes = [50, 100, 150, 200]
tpr = [0.17, 0.094, 0.066, 0.056]
# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, tpr, marker='o', linestyle='-', label='fpr')

# Labeling the axes
plt.xlabel('Sample size')
plt.ylabel('TPR')
plt.ylim(0, 1.0)
plt.savefig('TPRs.png')

# Adding a legend
# plt.legend()

# Display the plot
plt.show()