import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from match_ids import *
from scipy.signal import savgol_filter, argrelextrema
from scipy.interpolate import lagrange
from lmfit import minimize, Parameters, report_fit, Minimizer

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
all_observation_fields = np.unique(observation_fields)

selected_observation_fields = all_observation_fields
get_fields = len(selected_observation_fields)

# # V magnitude estimation
# V_jk = (galah_general['k_tmass'] + 2.*(galah_general['j_tmass'] - galah_general['k_tmass'] + 0.14) +
#         0.382 * np.exp((galah_general['j_tmass'] - galah_general['k_tmass'] - 0.2)/0.5))
#
# # select appropriate fields for the investigation
# # first filter them by date
# selected_observation_fields = all_observation_fields[all_observation_fields > 140203000000]
# # determine spread of RV values in the field
# n_selected = len(selected_observation_fields)
# rv_fields_span = np.ndarray(n_selected)
# rv_fields_std = np.ndarray(n_selected)
# V_jk_fields_mean = np.ndarray(n_selected)
# for i_f in range(n_selected):
#     field_rows = np.where(selected_observation_fields[i_f] == observation_fields)
#     V_jk_fields_mean[i_f] = np.nanmedian(V_jk[field_rows])
#     if V_jk_fields_mean[i_f] < 14 and len(field_rows[0]) > 75:
#         rv_data = galah_param[field_rows]['rv_guess']
#         rv_fields_span[i_f] = np.nanmax(rv_data) - np.nanmin(rv_data)
#         rv_fields_std[i_f] = np.nanstd(rv_data)
#     else:
#         rv_fields_span[i_f] = np.nan
#         rv_fields_std[i_f] = np.nan
# # rv_sort = (np.argsort(rv_fields_std)[np.isfinite(np.sort(rv_fields_std))])[::-1]
# rv_sort = (np.argsort(rv_fields_span)[np.isfinite(np.sort(rv_fields_span))])[::-1]
# get_fields = 40
# selected_observation_fields = selected_observation_fields[rv_sort[:get_fields]]  #select first few fields with the largest std
# # selected_observation_fields = [140209004201]

C_LIGHT = 299792458  # m/s

# line parameters

# Ca lines in red part of spectra
line_list = [6493.7810, 6499.6500, 6508.8496]
wvl_line_center = 6515

# Li line in red
# line_list = [6707.7635]
# wvl_line_center = 6707

# Na lines in green
# line_list = [5682.6333, 5688.2050]
# wvl_line_center = 5685

# Al lines in red
# line_list = [6696.0230, 6698.6730]
# wvl_line_center = 6697

# plot range and spectra reading control
# plot_range = 20.
# wvl_min = 1.

wvl_min = 6479
wvl_max = 6510

shift_for_barycentric = True

print 'Reading resampled GALAH spectra'
spectra_file_csv_obs = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_observed.csv'
spectra_file_csv_res = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv_obs)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_values = csv_param.get_wvl_values()

# determine the data range to be read and read it
# idx_read = np.logical_and(wvl_values > 6480,
#                           wvl_values < 6720)
idx_read = np.where(np.logical_and(wvl_values > wvl_min,
                                   wvl_values < wvl_max))
# spectral_data = np.loadtxt(galah_data_dir + spectra_file_csv, delimiter=',',
#                            usecols=np.arange(len(wvl_values))[idx_read])  # read limited number of columns instead of full dataset
# alternative and much faster way
spectral_data = pd.read_csv(galah_data_dir + spectra_file_csv_obs,
                            sep=',', header=None, na_values='nan', usecols=idx_read[0]).values
spectral_data_res = pd.read_csv(galah_data_dir + spectra_file_csv_res,
                            sep=',', header=None, na_values='nan', usecols=idx_read[0]).values
spectal_data_size = np.shape(spectral_data)
print spectal_data_size
wvl_read = wvl_values[idx_read]

out_dir = 'Telurics_test'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
os.chdir(out_dir)

prefix = 'tellurics'
tellurics_wvl_txt = prefix+'_wvl_all.txt'
observed_fields_spectra = 'fields_aggregated_spectra.csv'
if not os.path.isfile(tellurics_wvl_txt):
    txt_agg = open(observed_fields_spectra, 'w')
    txt_agg.close()
    tellurics = list([])
    for field_id in selected_observation_fields:  # filter by date
        print 'Step1 - working on field '+str(field_id)
        spectra_row = np.where(field_id == observation_fields)
        # rv values of every object in the field
        rv_field = galah_param[spectra_row]['rv_guess']
        # rv_range = np.nanmax(rv_field)-np.nanmin(rv_field)
        # rv_std = np.nanstd(rv_field)
        # print 'RV range: '+str(rv_range)
        # print 'RV spread: '+str(rv_std)
        # if rv_range < 50 or rv_std < 20:
        #     get_fields -= 1
        #     continue
        rv_weight = np.abs(rv_field)
        rv_weight = rv_weight/np.nanmax(rv_weight)
        max_weight = 0.2
        rv_weight = (1. - max_weight) + rv_weight * max_weight
        # median normalization of data in this observation field
        field_median_flux = np.nanmedian(spectral_data[spectra_row])
        # initialize plot
        fig, axes = plt.subplots(3, 1)
        for row in spectra_row[0]:
            # do a median correction for this spectra
            row_data = spectral_data[row, :]
            row_median = np.nanmedian(row_data)
            # row_data += field_median_flux - row_median
            # # store corrected spectra back
            # spectral_data[row, :] = row_data
            # plot this spectra
            axes[0].plot(wvl_read, row_data, color='blue', alpha=0.02, linewidth=0.8)
        spectral_data_field = spectral_data[spectra_row[0], :]
        spectral_data_avg = np.nanpercentile(spectral_data_field, 90, axis=0)
        axes[0].plot(wvl_read, spectral_data_avg, color='black', linewidth=0.75)
        # sigma clipping with average data
        # TODO if needed
        # signal filtering
        # remove nan values at the end or beginning of the spectra
        # idx_valid_spectra = np.isfinite(spectral_data_avg)
        # # data filtering
        # savgol_input = np.array(spectral_data_avg[idx_valid_spectra])
        # savgol_temp = savgol_filter(savgol_input, 7, 2)
        # spectral_data_avg_savgol = np.ndarray(len(wvl_read))
        # spectral_data_avg_savgol.fill(np.nan)
        # spectral_data_avg_savgol[idx_valid_spectra] = savgol_temp
        # axes[0].plot(wvl_read, spectral_data_avg_savgol, color='red', linewidth=0.75)
        axes[0].plot(wvl_read, np.average(spectral_data_field, weights=rv_weight, axis=0), color='red', linewidth=0.75)
        # save aggregated spectra
        aggregated_spectra = spectral_data_avg
        txt_agg = open(observed_fields_spectra, 'a')
        txt_agg.write(','.join([str(f) for f in aggregated_spectra])+'\n')
        txt_agg.close()
        # find telluric abs lines
        residuals = np.abs(spectral_data_avg - np.median(spectral_data_avg))
        idx_tellurics = argrelextrema(residuals, np.greater, order=3)
        tellurics.append(idx_tellurics)
        # add detected lines to the plot
        for abs_line in wvl_read[idx_tellurics]:
            axes[0].axvline(x=abs_line, color='black', linewidth=0.6)
        axes[0].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2), ylabel='Original spectra')
        #
        for row in spectra_row[0]:
            velocity_shift = galah_param[row]['rv_guess_shift']
            if shift_for_barycentric:
                velocity_shift -= galah_param[row]['v_bary']
                # barycentric_vel = galah_barycentric[galah_barycentric['sobject_id']==galah_param[row]['sobject_id']]['vel_ccd3'].data
                # velocity_shift -= barycentric_vel
            axes[1].plot(wvl_read/(1+velocity_shift*1000./C_LIGHT), spectral_data[row, :], color='blue', alpha=0.01, linewidth=0.8)
        for abs_line in line_list:
            axes[1].axvline(x=abs_line, color='black', linewidth=0.6)
        axes[1].set(xlim=(wvl_min, wvl_max), ylim=(0.4, 1.2), ylabel='RV shifted', xlabel='Wavelength')
        #
        spectra_res = spectral_data_res[spectra_row]
        spectra_res_median = np.nanmedian(spectra_res, axis=0)
        axes[1].plot(wvl_read, np.nanmean(spectra_res, axis=0), color='red', linewidth=0.6)
        axes[1].plot(wvl_read, spectra_res_median, color='green', linewidth=0.6)
        #
        for row in spectra_row[0]:
            velocity_shift = galah_param[row]['rv_guess_shift']
            velocity_shift -= galah_param[row]['v_bary']
            axes[2].plot(wvl_read * (1 + velocity_shift * 1000. / C_LIGHT), spectral_data_res[row, :]-spectra_res_median, color='blue', alpha=0.02,linewidth=0.8)
        axes[2].set(xlim=(wvl_min, wvl_max), ylim=(-0.2, 0.2), ylabel='RV shifted', xlabel='Wavelength')
        plt.tight_layout()
        plt.savefig(str(field_id)+'_1.png', dpi=200)
        plt.close()
    # write results
    tellurics_joined = np.array(np.hstack(tellurics)).flatten()
    print tellurics_joined
    str_tellurics = ','.join([str(l) for l in tellurics_joined])
    txt = open(tellurics_wvl_txt, 'w')
    txt.write(str_tellurics)
    txt.close()

tellurics_wvl_txt_2 = prefix+'_wvl.txt'
if not os.path.isfile(tellurics_wvl_txt_2):
    tellurics = np.loadtxt(tellurics_wvl_txt, delimiter=',')
    # determine telluric lines from observed minima of functions
    counts, position = np.histogram(tellurics, bins=len(wvl_read), range=(0, len(wvl_read)))
    tellurics_selected_1 = 100.*counts/get_fields > 33.
    # next method
    counts_local = np.convolve(counts, np.ones(3), mode='same')
    tellurics_local = 100.*counts_local/get_fields > 50.
    # at least 3 near observations
    tellurics_selected_2 = np.convolve(tellurics_local, np.ones(3), mode='same') == 3

    tellurics_selected_final = np.logical_or(tellurics_selected_1, tellurics_selected_2)
    # check for multiple detections at neighbouring wavelengths
    tellurics_selected_final_multi = np.convolve(tellurics_selected_final, np.ones(2), mode='same')
    tellurics_selected_final[tellurics_selected_final_multi >= 2] = False

    for abs_line in wvl_read[tellurics_selected_final]:
        plt.axvline(x=abs_line, color='red', linewidth=0.6)
    plt.bar(wvl_read, counts, align='center', width=0.06, linewidth=0)
    plt.xlim = (wvl_min, wvl_max)
    plt.savefig('tellurics_hist_2.png', dpi=200)
    plt.close()
    # write results
    str_tellurics = ','.join([str(l) for l in wvl_read[tellurics_selected_final]])
    txt = open(tellurics_wvl_txt_2, 'w')
    txt.write(str_tellurics)
    txt.close()

raise SystemExit

tellurics_param_txt = prefix+'_param.txt'
if not os.path.isfile(tellurics_param_txt):
    txt_file = open(tellurics_param_txt, 'w')
    txt_file.close()

    # function to be minimized
    def gaussian_fit(parameters, data, wvls, continuum, evaluate=True):
        n_keys = (len(parameters)) / 3
        # function_val = parameters['offset']*np.ones(len(wvls))
        function_val = np.array(continuum)
        for i_k in range(n_keys):
            function_val -= parameters['amp'+str(i_k)] * np.exp(-0.5 * (parameters['wvl'+str(i_k)] - wvls) ** 2 / parameters['std'+str(i_k)])
        if evaluate:
            # likelihood = np.nansum(np.power(data - function_val, 2))
            likelihood = np.power(data - function_val, 2)
            return likelihood
        else:
            return function_val

    telluric_lines = np.loadtxt(tellurics_wvl_txt_2, delimiter=',')
    for field_id in selected_observation_fields:  # filter by date
        print 'Step2 - working on field '+str(field_id)
        fig, axes = plt.subplots(2, 1)
        spectra_row = np.where(field_id == observation_fields)
        spectral_data_field = spectral_data[spectra_row[0], :]
        spectral_data_avg = np.nanpercentile(spectral_data_field, 90, axis=0)
        idx_valid_spectra = np.isfinite(spectral_data_avg)
        # select only non nan values for further analysis
        spectral_data_avg = spectral_data_avg[idx_valid_spectra]
        wvl_used = wvl_read[idx_valid_spectra]
        # data filtering
        spectral_data_avg_savgol = savgol_filter(spectral_data_avg, 7, 2)
        axes[0].plot(wvl_used, spectral_data_avg_savgol, color='black', linewidth=0.5)

        # continuum fit
        print 'Continuum fit'
        axes[1].plot(wvl_used, spectral_data_avg_savgol, color='black', linewidth=0.75)
        initial_chb_coef = np.polynomial.chebyshev.chebfit(wvl_used, spectral_data_avg_savgol, 3)
        cont_fit = np.polynomial.chebyshev.chebval(wvl_used, initial_chb_coef)
        axes[1].plot(wvl_used, cont_fit, color='red', linewidth=0.5)
        axes[1].set(ylim=(0.4, 1.2), xlim=(wvl_min, wvl_max))
        # perform sigma clipping before the next fitting cycle
        y_fit = np.array(cont_fit)
        y_ref = np.array(spectral_data_avg_savgol)
        x_fit = np.array(wvl_used)
        for i_f in range(8):
            data_diff = y_ref - y_fit
            idx_use_fit = np.abs(data_diff) < np.nanstd(data_diff)*2.
            print 'Using '+str(np.sum(idx_use_fit))+' of total '+str(len(wvl_used))+' points'
            # select data that will be fitted
            y_ref = y_ref[idx_use_fit]
            x_fit = x_fit[idx_use_fit]
            # chebyshev polinominal fit
            chb_coef = np.polynomial.chebyshev.chebfit(x_fit, y_ref, 12)
            y_fit = np.polynomial.chebyshev.chebval(x_fit, chb_coef)
            cont_fit = np.polynomial.chebyshev.chebval(wvl_used, chb_coef)
            # lagrange polynominal fit
            # lag_coef = lagrange(x_fit, y_ref)
            # lag_coef = np.poly1d(lag_coef.coeffs[-15:])
            # lang_fit = lag_coef(wvl_used)
            axes[1].plot(wvl_used, cont_fit, color='blue', linewidth=0.5)
            # axes[1].plot(wvl_used, lang_fit, color='pink', linewidth=0.5)
        for abs_line in telluric_lines:
            axes[1].axvline(x=abs_line, color='green', linewidth=0.3)

        # telluric lines fit
        print 'Fitting telluric lines'
        # determine parameters for a comb of gaussian profiles that will be fitted to the spectra
        fit_param = Parameters()
        # fit_param.add('offset', value=1., min=0.95, max=1.05)
        # for every line add std and amplitude of the absorption line
        fit_keys = list([])
        for i_l in range(len(telluric_lines)):
            key_std = 'std' + str(i_l)
            fit_param.add(key_std, value=0.01, min=0.001, max=0.1)
            fit_keys.append(key_std)
            key_amp = 'amp' + str(i_l)
            fit_param.add(key_amp, value=0.1, min=0.001, max=0.8)
            fit_keys.append(key_amp)
            key_wvl = 'wvl' + str(i_l)
            fit_param.add(key_wvl, value=telluric_lines[i_l], min=telluric_lines[i_l]-0.1, max=telluric_lines[i_l]+0.1)#, vary=False)
            fit_keys.append(key_wvl)
        # minimize the model
        fit_res = minimize(gaussian_fit, fit_param, #method='brute',
                           args=(spectral_data_avg_savgol, wvl_used, cont_fit))
                           # **{'max_nfev': 20000, 'verbose': 1})
        # minim = Minimizer(gaussian_fit, fit_param, fcn_args=(spectral_data_avg_savgol, wvl_used))
        # fit_res = minim.emcee(steps=100, nwalkers=100, ntemps=1)
        # get final parameters
        fit_res.params.pretty_print()
        report_fit(fit_res)
        fitted_curve = gaussian_fit(fit_res.params, 0., wvl_used, cont_fit, evaluate=False)
        fit_values_final = [p.value for p in fit_res.params.values()]
        fit_keys_final = [p.name for p in fit_res.params.values()]
        # save parameters
        txt_file = open(tellurics_param_txt, 'a')
        if os.path.getsize(tellurics_param_txt) == 0:
            txt_file.write(','.join(fit_keys_final)+'\n')
        txt_file.write(','.join([str(v) for v in fit_values_final])+'\n')
        txt_file.close()

        # plot the fit
        axes[0].plot(wvl_used, fitted_curve, color='red', linewidth=0.75)
        axes[0].set(ylim=(0.4, 1.2), xlim=(wvl_min, wvl_max))
        plt.savefig(str(field_id) + '_2.png', dpi=200)
        plt.close()

