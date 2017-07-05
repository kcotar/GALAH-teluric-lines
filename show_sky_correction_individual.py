import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from match_ids import *
from astropy.table import Table

imp.load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
from helper_functions import get_spectra_dr52

spectra_dir = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')

os.chdir('Sky_correction_individual_objects')
ccd = 3
for comb_id in list([1605220066, 1406090049, 1605200026, 1605220026, 1605220051]):
    # get runs used for this combined spectra
    runs = match_cobid_with_runid(comb_id, galah_observations)
    # get all objects in this filed
    idx_in_field = np.int64(galah_param['sobject_id'].data/1e5) == comb_id
    for idx_object in np.where(idx_in_field)[0]:
        sobject_id = galah_param[idx_object]['sobject_id']
        i_r = 0
        for run in runs:
            individual_id = combined_to_individual(sobject_id, run)
            spectra_0, _ = get_spectra_dr52(str(individual_id), root=spectra_dir, bands=[ccd], extension=0, individual=True)
            if len(spectra_0) == 0:
                continue
            spectra_2, wavelengths = get_spectra_dr52(str(individual_id), root=spectra_dir, bands=[ccd], extension=2, individual=True)
            if i_r == 0:
                spectra_diff = np.zeros_like(spectra_0[0])
            spectra_diff += (spectra_2[0] - spectra_0[0])
            i_r += 1
            plt.plot(wavelengths[0], spectra_2[0] - spectra_0[0], linewidth=0.6)
        plt.plot(wavelengths[0], spectra_diff/i_r, linewidth=0.3, color='black')
        plt.ylim((-100, 100))
        plt.xlim((6479, 6510))
        plt.savefig('{0}_{1}.png'.format(ccd, sobject_id), dpi=450)
        plt.close()