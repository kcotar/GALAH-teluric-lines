import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

dir = '/home/klemen/GALAH_data/Diagnostics/'
files = glob(dir+'')

for file in files:
    data = np.transpose(pd.read_csv(dir+file, sep=' ', header=None).values)
    plt.plot(data[0], data[3])
    # plt.plot(data[0], data[1], color='green')
    # plt.plot(data[0], data[1]/data[3])
plt.show()