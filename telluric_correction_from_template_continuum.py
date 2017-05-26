import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, splrep, splev
from astropy.table import Table
from match_ids import *

imp.load_source('s_helper', '../Carbon-Spectra/helper_functions.py')
from s_helper import get_spectra_dr52

imp.load_source('templates', '../H-alpha_map/template_spectra_function.py')
from templates import *


# Functions
def spectra_resample(spectra, wvl_orig, wvl_target, bspline=True):
    idx_finite = np.isfinite(spectra)
    min_wvl_s = np.nanmin(wvl_orig[idx_finite])
    max_wvl_s = np.nanmax(wvl_orig[idx_finite])
    idx_target = np.logical_and(wvl_target >= min_wvl_s,
                                wvl_target <= max_wvl_s)
    if bspline:
        bspline = splrep(wvl_orig[idx_finite], spectra[idx_finite])    #
        new_flux = splev(wvl_target[idx_target], bspline)
    else:
        func = interp1d(wvl_orig[idx_finite], spectra[idx_finite], assume_sorted=True, kind='linear')
        new_flux = func(wvl_target[idx_target])
    nex_flux_out = np.ndarray(len(wvl_target))
    nex_flux_out.fill(np.nan)
    nex_flux_out[idx_target] = new_flux
    return nex_flux_out


def wvl_values_range_lines(lines, wvl, width=1.):
    idx_pos = np.full_like(wvl, False)
    for line in lines:
        idx_pos = np.logical_or(idx_pos,
                                np.abs(wvl - line) <= width/2.)
    return idx_pos


def new_txt_file(filename):
    temp = open(filename, 'w')
    temp.close()


def append_line(filename, line_string, new_line=False):
    temp = open(filename, 'a')
    if new_line:
        temp.write(line_string+'\n')
    else:
        temp.write(line_string)
    temp.close()


print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_template_dir = '/home/klemen/GALAH_data/Spectra_template_grid/galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe/Teff_250_logg_0.50_feh_0.25_snr_40_medianshift_std_2.5/'
galah_spectra_data_dir52 = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
all_observation_fields = np.unique(observation_fields)

selected_observation_fields = all_observation_fields
get_fields = len(selected_observation_fields)

C_LIGHT = 299792458  # m/s

wvl_min = 6479
wvl_max = 6510
wvl_step = 0.02
wvl_observed = np.arange(wvl_min, wvl_max, wvl_step)

# atmospheric features line list
teluric_line_list = pd.read_csv('telluric_linelist.csv')
emission_line_list = pd.read_csv('emission_linelist_2_used.csv')
# subset of atmospheric lines
line_cor_width = 1.5  # Angstrom
teluric_line_list = teluric_line_list[np.logical_and(teluric_line_list['Ang'] > wvl_min,
                                                     teluric_line_list['Ang'] < wvl_max)]
emission_line_list = emission_line_list[emission_line_list['Flux'] >= 1.]
emission_line_list = emission_line_list[np.logical_and(emission_line_list['Ang'] > wvl_min,
                                                       emission_line_list['Ang'] < wvl_max)]

# create a mask of telluric positions at observed frame
idx_telluric = wvl_values_range_lines(teluric_line_list['Ang'], wvl_observed, width=1.5)

out_dir = 'Correction_with_template_continuum'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)

grid_list = Table.read(galah_template_dir + 'grid_list.csv', format='ascii.csv')
wvl_template = np.loadtxt(galah_template_dir + 'wvl_list.csv', delimiter=',')

# selected_observation_fields = list([140309002101])
out_fields = '_fields_list.csv'
out_continuum = '_template_continuum_median.csv'
if not os.path.isfile(out_continuum):
    new_txt_file(out_fields)
    new_txt_file(out_continuum)
    for field_id in selected_observation_fields:
        print 'Working on field ' + str(field_id)
        spectra_row = np.where(field_id == observation_fields)
        # initialize plot
        n_in_field = len(spectra_row[0])
        if n_in_field < 100:
            continue
        # create a stack of spectra residuals at continuum
        field_spectra_at_continuum = np.ndarray((n_in_field, len(wvl_observed)))
        field_spectra_at_continuum.fill(np.nan)
        fig, axes = plt.subplots(2, 1)
        for i_row in range(len(spectra_row[0])):
            row = spectra_row[0][i_row]
            object_param = galah_param[row]
            object_id = object_param['sobject_id']
            try:
                # get object spectra
                spectra_object, wvl_object = get_spectra_dr52(str(object_id), bands=[3], root=galah_spectra_data_dir52, extension=4)
                # get template spectra
                template_file = get_best_match(object_param['teff_guess'], object_param['logg_guess'], object_param['feh_guess'], grid_list, midpoint=False) + '.csv'
                spectra_template = np.loadtxt(galah_template_dir + template_file, delimiter=',')
                # resample them both to the final resolution
                velocity_shift = object_param['rv_guess_shift'] - object_param['v_bary']
                wvl_scale = (1 + velocity_shift * 1000. / C_LIGHT)
                spectra_template = spectra_resample(spectra_template, wvl_template * wvl_scale, wvl_observed)
                spectra_object = spectra_resample(spectra_object[0], wvl_object[0] * wvl_scale, wvl_observed)

                # determine continuum pixel positions that are located in the same position as observed telluric lines
                idx_spectra_use = np.logical_and(np.abs(1. - spectra_template) < 0.01,
                                                 idx_telluric)

                if np.sum(idx_spectra_use) > 0:
                    field_spectra_at_continuum[i_row, idx_spectra_use] = spectra_object[idx_spectra_use] - 1.
                    axes[0].plot(wvl_observed, field_spectra_at_continuum[i_row], color='blue', alpha=0.02, linewidth=0.8)
            except:
                print 'Something wrong with '+str(object_id)
        field_spectra_at_continuum_median = np.nanmedian(field_spectra_at_continuum, axis=0)
        axes[0].plot(wvl_observed, field_spectra_at_continuum_median, color='black', alpha=0.8, linewidth=0.8)
        axes[0].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Observed')
        axes[1].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Restframe', xlabel='Wavelength')
        axes[0].grid(True)
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(str(field_id) + '_continuum_spectra.png', dpi=250)
        plt.close()
        # save results to files
        append_line(out_fields, str(field_id), new_line=True)
        append_line(out_continuum, ','.join([str(f) for f in field_spectra_at_continuum_median]), new_line=True)
