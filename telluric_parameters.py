import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table

os.chdir('Ca_lines-red_fit_complete')

telluric_data = Table.read('tellurics_red_param_old.txt', format='ascii.csv')

for i_t in range(len(telluric_data.colnames)/3):
    amp_values = telluric_data['amp'+str(i_t)]
    std_values = telluric_data['std'+str(i_t)]
    amp_median = np.median(amp_values)
    std_mean = np.mean(std_values[amp_values > amp_median])
    plt.scatter(amp_values, std_values)
    plt.axhline(y=std_mean, color='red')
    plt.axvline(x=amp_median, color='black')
    plt.xlabel('Amplitude')
    plt.ylabel('Standard deviation')
    plt.savefig('amp_std_line_{0}.png'.format(i_t))
    plt.close()