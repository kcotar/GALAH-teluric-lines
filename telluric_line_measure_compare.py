import os
import numpy as np
import matplotlib.pyplot as plt

out_dir = 'Line_measurements'
os.chdir(out_dir)

wvl_investigate1 = 6504.999  # Angstrom
out_txt1 = 'wvl_{0}_2.txt'.format(wvl_investigate1)
l1 = np.loadtxt(out_txt1, delimiter=',')

wvl_investigate2 = 6498.7368  # Angstrom
out_txt2 = 'wvl_{0}_2.txt'.format(wvl_investigate2)
l2 = np.loadtxt(out_txt2, delimiter=',')

idx = np.logical_and(l1[:,2] < 0.5, l2[:,2] < 0.5)

amp_fit = np.polyfit(l1[idx,1], l2[idx,1], 1)
print amp_fit
print 'std1:', np.median(l1[idx,2])
print 'std2:', np.median(l2[idx,2])

min=-500
max=500
x = np.arange(min, max, 50)
plt.scatter(l1[idx,1], l2[idx,1])
plt.plot(x, x*amp_fit[0] + amp_fit[1])
plt.xlim((min,max))
plt.savefig('correlation_x{0}_y{1}_k{2}.png'.format(wvl_investigate1, wvl_investigate2, amp_fit[0]), dpi=100)
plt.close()