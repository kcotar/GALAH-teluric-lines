mport os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from match_ids import *
from astropy.table import Table

imp.load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
from helper_functions import get_spectra_dr52

spectra_dir = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'galah_objects_complete.csv', format='ascii.csv')


os.chdir('Sky_spectra_corsstalk')
ccd = 3