import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from match_ids import *
from template_spectra_function import *

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_template_dir = '/home/klemen/GALAH_data/Spectra_template_grid/galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe/Teff_250_logg_0.50_feh_0.25_snr_ 40_medianshift_std_2.5/'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
all_observation_fields = np.unique(observation_fields)

selected_observation_fields = all_observation_fields
get_fields = len(selected_observation_fields)


C_LIGHT = 299792458  # m/s

wvl_min = 6479
wvl_max = 6510

shift_for_barycentric = True

print 'Reading resampled GALAH spectra'
spectra_file_csv_obs = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv_obs)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_values = csv_param.get_wvl_values()

# determine the data range to be read and read it
idx_read = np.where(np.logical_and(wvl_values > wvl_min,
                                   wvl_values < wvl_max))
spectral_data = pd.read_csv(galah_data_dir + spectra_file_csv_obs,
                            sep=',', header=None, na_values='nan', usecols=idx_read[0]).values
spectal_data_size = np.shape(spectral_data)
print spectal_data_size
wvl_read = wvl_values[idx_read]

out_dir = 'Correction_with_template'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
os.chdir(out_dir)

grid_list = Table.read(galah_template_dir + 'grid_list.csv', format='ascii.csv')
# selected_observation_fields = list([160402006601])
for field_id in selected_observation_fields:  # filter by date
    print 'Working on field '+str(field_id)
    spectra_row = np.where(field_id == observation_fields)
    # initialize plot
    if len(spectra_row[0]) < 25:
        continue
    fig, axes = plt.subplots(2, 1)
    for row in spectra_row[0]:
        object_param = galah_param[row]
        object_spectra = spectral_data[row]
        # get template spectra
        template_file = get_best_match(object_param['teff_guess'], object_param['logg_guess'], object_param['feh_guess'], grid_list, midpoint=False)+'.csv'
        template_spectra = np.loadtxt(galah_template_dir + template_file, delimiter=',')[idx_read]
        # subtract spectra
        spectra_residuum = object_spectra - template_spectra
        # plt.plot(wvl_read, template_spectra, color='black')
        # plt.plot(wvl_read, object_spectra, color='blue')
        # plt.show()
        # shift to observed frame
        velocity_shift = object_param['rv_guess_shift']
        velocity_shift -= object_param['v_bary']
        wvl_shifted = wvl_read * (1 + velocity_shift * 1000. / C_LIGHT)
        # plot graphs
        axes[0].plot(wvl_shifted, spectra_residuum, color='blue', alpha=0.02, linewidth=0.8)
        axes[1].plot(wvl_read, spectra_residuum, color='blue', alpha=0.02, linewidth=0.8)
    axes[0].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Observed')
    axes[1].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Restframe', xlabel='Wavelength')
    plt.tight_layout()
    plt.savefig(str(field_id)+'.png', dpi=250)
    plt.close()
