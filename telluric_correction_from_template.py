import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, splrep, splev
from astropy.table import Table
from match_ids import *
from lmfit import Parameters, minimize

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

imp.load_source('templates', '../H-alpha_map/template_spectra_function.py')
from templates import *


# Functions
def spectra_resample(spectra, wvl_orig, wvl_target):
    idx_finite = np.isfinite(spectra)
    min_wvl_s = np.nanmin(wvl_orig[idx_finite])
    max_wvl_s = np.nanmax(wvl_orig[idx_finite])
    bspline = splrep(wvl_orig[idx_finite], spectra[idx_finite])
    idx_target = np.logical_and(wvl_target >= min_wvl_s,
                                wvl_target <= max_wvl_s)
    new_flux = splev(wvl_target[idx_target], bspline)
    nex_flux_out = np.ndarray(len(wvl_target))
    nex_flux_out.fill(np.nan)
    nex_flux_out[idx_target] = new_flux
    return nex_flux_out


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


def minimize_scale(param, template_residuum, teluric):
    param_values = param.valuesdict()
    return np.power(template_residuum + teluric*param_values['scale'], 2)


def wvl_values_range_lines(lines, wvl, width=1.):
    idx_pos = np.full_like(wvl, False)
    for line in lines:
        idx_pos = np.logical_or(idx_pos,
                                np.abs(wvl - line) <= width/2.)
    return idx_pos


def extrema_value(vals):
    min = np.nanmin(vals)
    max = np.nanmax(vals)
    if np.abs(min) > np.abs(max):
        return min
    else:
        return max

print 'Reading data sets'
galah_template_dir = '/home/klemen/GALAH_data/Spectra_template_grid/galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe/Teff_250_logg_0.50_feh_0.25_snr_40_medianshift_std_2.5/'
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
spectral_data_size = np.shape(spectral_data)
print spectral_data_size
wvl_read = wvl_values[idx_read]
wvl_read_finer = np.arange(wvl_min, wvl_max, csv_param.get_wvl_step()/3)

# atmospheric features line list
teluric_line_list = pd.read_csv('telluric_linelist.csv')
emission_line_list = pd.read_csv('emission_linelist.csv')
# subset of atmospheric lines
line_cor_width = 1.5  # Angstrom
teluric_line_list = teluric_line_list[np.logical_and(teluric_line_list['Ang'] > wvl_min,
                                                     teluric_line_list['Ang'] < wvl_max)]
emission_line_list = emission_line_list[np.logical_and(emission_line_list['Ang'] > wvl_min,
                                                       emission_line_list['Ang'] < wvl_max)]

out_dir = 'Correction_with_template'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)

grid_list = Table.read(galah_template_dir + 'grid_list.csv', format='ascii.csv')
# selected_observation_fields = list([170205004401])
out_fields = '_fields_list.csv'
out_residuals = '_template_residuals_median.csv'
out_residuals_abs = '_template_residuals_median_abs.csv'
if not os.path.isfile(out_residuals):
    new_txt_file(out_fields)
    new_txt_file(out_residuals)
    new_txt_file(out_residuals_abs)
    for field_id in selected_observation_fields:  # filter by date
        print 'Working on field '+str(field_id)
        spectra_row = np.where(field_id == observation_fields)
        # initialize plot
        n_in_field = len(spectra_row[0])
        if n_in_field < 100:
            continue
        # create a stack of field residuals
        field_residuals = np.ndarray((n_in_field, len(wvl_read_finer)))
        # init plot
        fig, axes = plt.subplots(2, 1)
        for i_row in range(len(spectra_row[0])):
            row = spectra_row[0][i_row]
            object_param = galah_param[row]
            object_spectra = spectral_data[row]
            # get template spectra
            template_file = get_best_match(object_param['teff_guess'], object_param['logg_guess'], object_param['feh_guess'], grid_list, midpoint=False)+'.csv'
            template_spectra = np.loadtxt(galah_template_dir + template_file, delimiter=',')[idx_read]
            # subtract spectra
            spectra_residuum = object_spectra - template_spectra
            # shift to observed frame
            velocity_shift = object_param['rv_guess_shift'] - object_param['v_bary']
            wvl_shifted = wvl_read * (1 + velocity_shift * 1000. / C_LIGHT)
            # resample shifted RV/barycentric shifted to common positions - OVERSAMPLING DATA
            spectra_residuum_resampled = spectra_resample(spectra_residuum, wvl_shifted, wvl_read_finer)  # spectrum, in, out
            field_residuals[i_row, :] = spectra_residuum_resampled
            # plot graphs
            axes[0].plot(wvl_shifted, spectra_residuum, color='blue', alpha=0.02, linewidth=0.8)
            axes[1].plot(wvl_read, spectra_residuum, color='blue', alpha=0.02, linewidth=0.8)
            # emission investigation
            # idx_emission_1 = wvl_values_range_lines(list([emission_line_list['Ang'].values[0]]), wvl_read_finer, width=1.5)
            # idx_emission_2 = wvl_values_range_lines(list([emission_line_list['Ang'].values[0]]), wvl_read_finer, width=1.5)
            # print extrema_value(spectra_residuum_resampled[idx_emission_1]), extrema_value(spectra_residuum_resampled[idx_emission_2])
        # residuals statistics
        # determine a first telluric approximation for all objects in a field
        residuum_median = np.nanmedian(field_residuals, axis=0)
        residuum_median_abs = np.nanmedian(np.abs(field_residuals), axis=0)
        axes[0].plot(wvl_read_finer, residuum_median, color='black', alpha=0.8, linewidth=0.8)
        axes[0].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Observed')
        axes[1].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Restframe', xlabel='Wavelength')
        plt.tight_layout()
        plt.savefig(str(field_id)+'.png', dpi=250)
        plt.close()
        # save results to files
        append_line(out_fields, str(field_id), new_line=True)
        append_line(out_residuals, ','.join([str(f) for f in residuum_median]), new_line=True)
        append_line(out_residuals_abs, ','.join([str(f) for f in residuum_median_abs]), new_line=True)

# --------------------------------------------------
# READ DATA from previous processing step
fields_ids = pd.read_csv(out_fields, header=None, sep=',', na_values='nan').values.ravel()
fields_residuals = pd.read_csv(out_residuals, header=None, sep=',', na_values='nan').values
fields_residuals_abs = pd.read_csv(out_residuals_abs, header=None, sep=',', na_values='nan').values
# --------------------------------------------------

# identification of telluric wavelengths or structures
mean_field_residuals = np.nanmean(fields_residuals, axis=0)
idx_telluric_wvl = wvl_values_range_lines(teluric_line_list['Ang'], wvl_read_finer, width=1.5)
for i_f in range(fields_residuals.shape[0]):
    plt.plot(wvl_read_finer, fields_residuals[i_f], color='blue', alpha=0.02, linewidth=0.8)
plt.scatter(wvl_read_finer[idx_telluric_wvl], mean_field_residuals[idx_telluric_wvl], s=3, c='black', lw=0)
for line in teluric_line_list['Ang'].values:
    plt.axvline(x=line, color='red', linewidth=0.5)
for line in emission_line_list['Ang'].values:
    plt.axvline(x=line, color='green', linewidth=0.5)
plt.xlim((wvl_min, wvl_max))
plt.ylim((-0.2, 0.2))
plt.savefig('_template_residuals.png', dpi=350)
plt.close()

# identification of emission wavelengths or structures
mean_field_residuals_abs = np.nanmean(fields_residuals_abs, axis=0)
idx_emission_wvl = wvl_values_range_lines(emission_line_list['Ang'], wvl_read_finer, width=1.5)
for i_f in range(fields_residuals_abs.shape[0]):
    plt.plot(wvl_read_finer, fields_residuals_abs[i_f], color='blue', alpha=0.02, linewidth=0.8)
plt.scatter(wvl_read_finer[idx_emission_wvl], mean_field_residuals_abs[idx_emission_wvl], s=3, c='black', lw=0)
for line in teluric_line_list['Ang'].values:
    plt.axvline(x=line, color='red', linewidth=0.5)
for line in emission_line_list['Ang'].values:
    plt.axvline(x=line, color='green', linewidth=0.5)
plt.xlim((wvl_min, wvl_max))
plt.ylim((-0.2, 0.2))
plt.savefig('_template_residuals_abs.png', dpi=350)
plt.close()

# refine this approximation using the information about the wavelengths of the telluric lines
# raise SystemExit

# subset of observations - for test corrections
i_obj_selected = np.where(observation_fields == 140117002601)[0]

# try to fit the approximation to every individual spectra in the field
for i_obj in i_obj_selected:  # range(0, len(galah_param), 25):
    object_param = galah_param[i_obj]
    object_id = object_param['sobject_id']
    object_in_field = np.int64(object_id/1000.)
    print 'Correcting object: '+str(object_id)+'  field: '+str(object_in_field)
    if object_in_field in fields_ids:
        # get previously determined residual
        field_residual_observed = fields_residuals[np.where(fields_ids == object_in_field)[0], :][0]
        field_residual_abs_observed = fields_residuals_abs[np.where(fields_ids == object_in_field)[0], :][0]
        object_spectra = spectral_data[i_obj]
        velocity_shift = object_param['rv_guess_shift'] - object_param['v_bary']

        # get template spectra
        template_file = get_best_match(object_param['teff_guess'], object_param['logg_guess'],
                                       object_param['feh_guess'], grid_list, midpoint=False) + '.csv'
        template_spectra = np.loadtxt(galah_template_dir + template_file, delimiter=',')[idx_read]

        # --------------------------------------------------
        # STEP 1:
        # wavelengths to be corrected - first correct part of the spectra influenced by the telluric effects
        idx_emission_wvl = wvl_values_range_lines(emission_line_list['Ang'], wvl_read_finer, width=1.)
        idx_telluric_wvl = wvl_values_range_lines(teluric_line_list['Ang'], wvl_read_finer, width=1.5)
        idx_correction_skip = np.logical_or(np.logical_not(idx_telluric_wvl), idx_emission_wvl)
        # create new temporary array from determined residuals
        field_residuum_telluric = np.array(field_residual_observed)
        # unset masked wavelengths
        field_residuum_telluric[idx_correction_skip] = 0
        # resample and shift residuals from observed to restframe
        field_residuum_telluric = spectra_resample(field_residuum_telluric, wvl_read_finer/(1 + velocity_shift * 1000. / C_LIGHT), wvl_read)
        field_residuum_telluric_binary = spectra_resample(np.logical_not(idx_correction_skip),
                                                          wvl_read_finer/(1 + velocity_shift * 1000. / C_LIGHT), wvl_read) > 0.5

        # set the residuum outside observed wavelengths to 0
        object_spectra_corrected_s1 = object_spectra - field_residuum_telluric
        template_res_before_s1 = template_spectra - object_spectra
        template_res_after_s1 = template_spectra - object_spectra_corrected_s1

        # the best residuum scaling factor based on reference spectra
        fit_param = Parameters()
        fit_param.add('scale', value=1, min=0., max=10.)
        fit_res = minimize(minimize_scale, fit_param,
                           args=(template_res_before_s1, field_residuum_telluric),
                           **{'nan_policy': 'omit'})
        fit_res.params.pretty_print()

        residuum_scale = fit_res.params['scale'].value
        object_spectra_corrected_s1_scaled = object_spectra - residuum_scale * field_residuum_telluric
        template_res_after_s1_scaled = template_spectra - object_spectra_corrected_s1_scaled

        fig, axes = plt.subplots(2, 1)
        suptitle = 'Guess  ->  teff:{:4.0f}  logg:{:1.1f}  feh:{:1.1f} \n correction scale:{:.1f}'.format(object_param['teff_guess'],
                                                                               object_param['logg_guess'],
                                                                               object_param['feh_guess'], residuum_scale)
        fig.suptitle(suptitle)
        axes[0].plot(wvl_read, template_spectra, color='red', linewidth=0.8)
        axes[0].plot(wvl_read, object_spectra, color='black', linewidth=0.8)
        axes[0].plot(wvl_read, object_spectra_corrected_s1, color='blue', linewidth=0.8)
        axes[0].plot(wvl_read, object_spectra_corrected_s1_scaled, color='green', linewidth=0.8)
        axes[0].set(xlim=(wvl_min, wvl_max), ylim=(0.2, 1.1), ylabel='Spectra')
        on = False
        off = True
        for i_t in range(len(field_residuum_telluric_binary)):
            f_r_b = field_residuum_telluric_binary[i_t]
            if off and f_r_b:
                span_s = i_t
                off = False
                on = True
            if on and not f_r_b:
                span_e = i_t
                off = True
                on = False
                axes[1].axvspan(wvl_read[span_s], wvl_read[span_e], facecolor='black', alpha=0.1)
        axes[1].plot(wvl_read, field_residuum_telluric + 0.1, color='black', linewidth=0.8)
        axes[1].plot(wvl_read, template_res_before_s1, color='black', linewidth=0.8)
        axes[1].plot(wvl_read, template_res_after_s1, color='blue', linewidth=0.8)
        axes[1].plot(wvl_read, template_res_after_s1_scaled, color='green', linewidth=0.8)
        axes[1].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Template residual')
        axes[1].grid(True)
        plt.savefig(str(object_id)+'_t.png', dpi=350)
        plt.close()
        # --------------------------------------------------

        # --------------------------------------------------
        # STEP 2:
        # wavelengths to be corrected - first correct part of the spectra influenced by the telluric effects
        # wavelengths to be corrected - first correct part of the spectra influenced by the telluric effects
        idx_emission_wvl = wvl_values_range_lines(emission_line_list['Ang'], wvl_read_finer, width=1.5)
        idx_telluric_wvl = wvl_values_range_lines(teluric_line_list['Ang'], wvl_read_finer, width=1.)
        idx_correction_skip = np.logical_or(np.logical_not(idx_emission_wvl), idx_telluric_wvl)
        # create new temporary array from determined residuals
        field_residuum_emission = np.array(field_residual_abs_observed)
        # unset masked wavelengths
        field_residuum_emission[idx_correction_skip] = 0
        # resample and shift residuals from observed to restframe
        field_residuum_emission = spectra_resample(field_residuum_emission,
                                                   wvl_read_finer / (1 + velocity_shift * 1000. / C_LIGHT), wvl_read)
        field_residuum_emission_binary = spectra_resample(np.logical_not(idx_correction_skip),
                                                          wvl_read_finer / (1 + velocity_shift * 1000. / C_LIGHT),
                                                          wvl_read) > 0.5

        # set the residuum outside observed wavelengths to 0
        object_spectra_corrected_s2 = object_spectra - field_residuum_emission
        template_res_before_s2 = template_spectra - object_spectra
        template_res_after_s2 = template_spectra - object_spectra_corrected_s2

    fig, axes = plt.subplots(2, 1)
    # suptitle = 'Guess  ->  teff:{:4.0f}  logg:{:1.1f}  feh:{:1.1f} \n correction scale:{:.1f}'.format(
    #     object_param['teff_guess'],
    #     object_param['logg_guess'],
    #     object_param['feh_guess'], residuum_scale)
    # fig.suptitle(suptitle)
    axes[0].plot(wvl_read, template_spectra, color='red', linewidth=0.8)
    axes[0].plot(wvl_read, object_spectra, color='black', linewidth=0.8)
    axes[0].plot(wvl_read, object_spectra_corrected_s2, color='blue', linewidth=0.8)
    # axes[0].plot(wvl_read, object_spectra_corrected_s1_scaled, color='green', linewidth=0.8)
    axes[0].set(xlim=(wvl_min, wvl_max), ylim=(0.2, 1.1), ylabel='Spectra')
    on = False
    off = True
    for i_t in range(len(field_residuum_emission_binary)):
        f_r_b = field_residuum_emission_binary[i_t]
        if off and f_r_b:
            span_s = i_t
            off = False
            on = True
        if on and not f_r_b:
            span_e = i_t
            off = True
            on = False
            axes[1].axvspan(wvl_read[span_s], wvl_read[span_e], facecolor='black', alpha=0.1)
    axes[1].plot(wvl_read, field_residuum_emission + 0.1, color='black', linewidth=0.8)
    axes[1].plot(wvl_read, template_res_before_s2, color='black', linewidth=0.8)
    axes[1].plot(wvl_read, template_res_after_s2, color='blue', linewidth=0.8)
    # axes[1].plot(wvl_read, template_res_after_scaled, color='green', linewidth=0.8)
    axes[1].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='Template residual')
    axes[1].grid(True)
    plt.savefig(str(object_id) + '_e.png', dpi=350)
    plt.close()

# refine the approximations using the position of the stars in the observation field and their altitude angle
