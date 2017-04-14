import os, imp
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from match_ids import *

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_objects = Table.read(galah_data_dir+'galah_objects.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)

C_LIGHT = 299792458  # m/s

# line parameters

# Ca lines in red part of spectra
# line_list =  [6493.7810, 6499.6500, 6508.8496]
# wvl_line_center = 6500

# Li line in red
# line_list = [6707.7635]
# wvl_line_center = 6707

# Na lines in green
line_list = [5682.6333, 5688.2050]
wvl_line_center = 5685

# Al lines in red
# line_list = [6696.0230, 6698.6730]
# wvl_line_center = 6697

# plot range and spectra reading control
plot_range = 11.
read_additional = 3.



print 'Reading resampled GALAH spectra'
spectra_file_csv = 'galah_dr51_ccd2_5660_5850_interpolated_wvlstep_0.05_linear.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_step = csv_param.get_wvl_step()
wvl_values = wvl_start + np.float64(range(0, np.int32(np.ceil((wvl_end-wvl_start)/wvl_step)))) * wvl_step

# determine the data range to be read and read it
idx_read = np.logical_and(wvl_values > (wvl_line_center - plot_range - read_additional),
                          wvl_values < (wvl_line_center + plot_range + read_additional))
spectral_data = np.loadtxt(galah_data_dir + spectra_file_csv, delimiter=',',
                           usecols=np.arange(len(wvl_values))[idx_read])  # read limited number of columns instead of full dataset
spectal_data_size = np.shape(spectral_data)
wvl_read = wvl_values[idx_read]

out_dir = 'Na_lines-green'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
os.chdir(out_dir)
for field_id in np.unique(observation_fields):
    print 'Working on field '+str(field_id)
    fig, axes = plt.subplots(2, 1)
    spectra_row = np.where(field_id == observation_fields)
    for row in spectra_row[0]:
        sobject_id = galah_param[row]['sobject_id']
        comb_id = get_id(sobject_id)
        run_ids = match_cobid_with_runid(comb_id, galah_observations)
        bar_vels = get_barycentric(sobject_id, run_ids, galah_objects, ccd='1')
        barycentric_rv = np.mean(bar_vels)
        axes[0].plot(wvl_read, spectral_data[row, :], color='blue', alpha=0.05, linewidth=0.8)
    axes[0].plot(wvl_read, np.nanmedian(spectral_data[spectra_row[0], :], axis=0), color='black', linewidth=0.8)
    axes[0].plot(wvl_read, np.nanmax(spectral_data[spectra_row[0], :], axis=0), color='black', linewidth=0.3)
    for abs_line in line_list:
        axes[0].axvline(x=abs_line, color='black', linewidth=0.6)
    axes[0].set(xlim=(wvl_line_center - plot_range, wvl_line_center + plot_range), ylim=(0.4, 1.2), ylabel='Original spectra')
    for row in spectra_row[0]:
        plt.plot(wvl_read/(1+galah_param[row]['rv_guess']*1000./C_LIGHT), spectral_data[row, :], color='blue', alpha=0.05, linewidth=0.8)
    for abs_line in line_list:
        axes[1].axvline(x=abs_line, color='black', linewidth=0.6)
    axes[1].set(xlim=(wvl_line_center - plot_range, wvl_line_center + plot_range), ylim=(0.4, 1.2), ylabel='RV shifted', xlabel='Wavelength')
    plt.tight_layout()
    plt.savefig(str(field_id)+'.png', dpi=200)
    plt.close()

