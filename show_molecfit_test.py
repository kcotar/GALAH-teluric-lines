import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = '/home/nandir/Desktop/GALAH_data/Diagnostics/'
files = ['01apr30018.txt','10nov30018.txt','11dec30016.txt','11nov30018.txt','11nov30019.txt','11nov30020.txt','11nov30023.txt','11nov30024.txt','11nov30025.txt','11nov30026.txt','11nov30027.txt','11nov30028.txt']

for file in files:
    data = np.transpose(pd.read_csv(dir+file, sep=' ', header=None).values)
    plt.plot(data[0], data[3])
    # plt.plot(data[0], data[1], color='green')
    # plt.plot(data[0], data[1]/data[3])
plt.show()