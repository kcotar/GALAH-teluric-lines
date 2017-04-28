import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/nandir/Desktop/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
all_observation_fields = np.unique(observation_fields)

selected_observation_fields = all_observation_fields
get_fields = len(selected_observation_fields)

C_LIGHT = 299792458  # m/s

wvl_min = 6479
wvl_max = 6520

print 'Reading resampled GALAH spectra'
spectra_file_csv = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_observed.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_values = csv_param.get_wvl_values()

# determine the data range to be read and read it
idx_read = np.where(np.logical_and(wvl_values > wvl_min,
                                   wvl_values < wvl_max))
# alternative and much faster way
spectral_data = pd.read_csv(galah_data_dir + spectra_file_csv,
                            sep=',', header=None, na_values='nan', usecols=idx_read[0]).values
spectal_data_size = np.shape(spectral_data)
print spectal_data_size
wvl_read = wvl_values[idx_read]

out_dir = 'Multiplots_observed'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
os.chdir(out_dir)

abs_lines = np.array([6480.1,6483.28,6486.82,6490.84,6492.94,6500.32,6504.22,6506.68,6508.6])

for field_id in selected_observation_fields:  # filter by date
    print 'Step1 - working on field ' + str(field_id)
    spectra_row = np.where(field_id == observation_fields)
    # order them by radial velocity
    rv_vels = galah_param['rv_guess'][spectra_row[0]]
    spectra_row = spectra_row[0][np.argsort(rv_vels)]

    print ' number od spectra in field is '+str(len(spectra_row))
    i_p = 1
    n_s = 0
    n_max = 15
    for row in spectra_row:
        spectra = spectral_data[row, :]
        plt.plot(wvl_read, spectra + n_s*0.2, linewidth=0.6, color='black', alpha=0.8)
        n_s += 1
        if n_s == 1:
            for abs_line in abs_lines:
                plt.axvline(x=abs_line, color='blue', linewidth=0.6)
        if n_s >= n_max:
            plt.xlim((wvl_min, wvl_max))
            plt.savefig(str(field_id)+'_{:02.0f}'.format(i_p)+'.png', dpi=300, )
            plt.close()
            i_p += 1
            n_s = 0
    plt.xlim((wvl_min, wvl_max))
    plt.savefig(str(field_id) + '_{:02.0f}'.format(i_p) + '.png', dpi=300)
    plt.close()

    # fig, axes = plt.subplots(2, 1)
    # for row in spectra_row[0]:
    #     rv_shift = galah_param[row]['rv_guess_shift']
    #     rv_guess = galah_param[row]['rv_guess']
    #     v_bary = galah_param[row]['v_bary']
    #     spectra = spectral_data[row, :]
    #     # axes[0].plot(wvl_read, spectra, color='blue', alpha=0.05, linewidth=0.4)
    #     # axes[1].plot(wvl_read*(1. + rv_shift*1000./C_LIGHT), spectra, color='blue', alpha=0.05, linewidth=0.4)
    #     axes[0].plot(wvl_read*(1. + (rv_shift-v_bary)*1000./C_LIGHT), spectra, color='blue', alpha=0.05, linewidth=0.4)
    #     axes[1].plot(wvl_read*(1. + (rv_guess-v_bary)*1000./C_LIGHT), spectra, color='blue', alpha=0.05, linewidth=0.4)
    # axes[0].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2))
    # axes[1].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2))
    # # axes[2].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2))
    # # axes[3].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2))
    # plt.savefig(str(field_id)+'.png', dpi=600)
    # plt.tight_layout()
    # plt.close()

