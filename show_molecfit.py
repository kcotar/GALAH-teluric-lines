import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/nandir/Desktop/GALAH_data/'
fields_param = pd.read_csv(galah_data_dir+'sobject_iraf_52_reduced_fields.csv', header=None, sep=',').values[0]
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field

print 'Reading resampled GALAH spectra'
molecfit_csv = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.02_spline_diagnostics.csv'
spectra_file_csv = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(molecfit_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_values = csv_param.get_wvl_values()
csv_param_2 = CollectionParameters(spectra_file_csv)
wvl_values_spectra = csv_param_2.get_wvl_values()

# determine the data range to be read and read it
# idx_read = np.where(np.logical_and(wvl_values > wvl_min,
#                                    wvl_values < wvl_max))
# spectral_data = np.loadtxt(galah_data_dir + spectra_file_csv, delimiter=',',
#                            usecols=np.arange(len(wvl_values))[idx_read])  # read limited number of columns instead of full dataset
# alternative and much faster way
molecfit_data = pd.read_csv(galah_data_dir + molecfit_csv,
                            sep=',', header=None, na_values='nan').values
# spectral_data = pd.read_csv(galah_data_dir + spectra_file_csv,
#                             sep=',', header=None, na_values='nan').values

field_id = np.int64(galah_param['sobject_id']/1000)
for i_f in range(len(fields_param)):
    molecfit = molecfit_data[i_f]
    if np.isfinite(molecfit).any():
        print fields_param[i_f]
        row_use = np.where(field_id == fields_param[i_f])[0]
        # plt.plot(wvl_values_spectra, np.nanmedian(spectral_data[row_use], axis=0), linewidth=0.6, color='blue')
        plt.plot(wvl_values, molecfit, linewidth=0.6)
plt.show()
