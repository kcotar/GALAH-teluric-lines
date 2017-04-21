import os, imp
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from match_ids import *
from scipy.signal import savgol_filter, argrelextrema
from lmfit import minimize, Parameters, report_fit, Minimizer

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')
galah_barycentric = Table.read(galah_data_dir+'galah_barycentric.csv')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
all_observation_fields = np.unique(observation_fields)

# V magnitude estimation
V_jk = (galah_general['k_tmass'] + 2.*(galah_general['j_tmass'] - galah_general['k_tmass'] + 0.14) +
        0.382 * np.exp((galah_general['j_tmass'] - galah_general['k_tmass'] - 0.2)/0.5))

# select appropriate fields for the investigation
# first filter them by date
selected_observation_fields = all_observation_fields[all_observation_fields > 140203000000]
# determine spread of RV values in the field
n_selected = len(selected_observation_fields)
rv_fields_span = np.ndarray(n_selected)
rv_fields_std = np.ndarray(n_selected)
V_jk_fields_mean = np.ndarray(n_selected)
for i_f in range(n_selected):
    field_rows = np.where(selected_observation_fields[i_f] == observation_fields)
    V_jk_fields_mean[i_f] = np.nanmedian(V_jk[field_rows])
    if V_jk_fields_mean[i_f] < 14 and len(field_rows[0]) > 75:
        rv_data = galah_param[field_rows]['rv_guess']
        rv_fields_span[i_f] = np.nanmax(rv_data) - np.nanmin(rv_data)
        rv_fields_std[i_f] = np.nanstd(rv_data)
    else:
        rv_fields_span[i_f] = np.nan
        rv_fields_std[i_f] = np.nan
# rv_sort = (np.argsort(rv_fields_std)[np.isfinite(np.sort(rv_fields_std))])[::-1]
rv_sort = (np.argsort(rv_fields_span)[np.isfinite(np.sort(rv_fields_span))])[::-1]
get_fields= 35
selected_observation_fields = selected_observation_fields[rv_sort[:get_fields]]  #select first few fields with the largest std
# selected_observation_fields = [140209004201]

C_LIGHT = 299792458  # m/s

# line parameters

# Ca lines in red part of spectra
line_list = [6493.7810, 6499.6500, 6508.8496]
wvl_line_center = 6510

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
plot_range = 10.
read_additional = 1.

shift_for_barycentric = True

print 'Reading resampled GALAH spectra'
spectra_file_csv = 'galah_dr51_ccd3_6490_6710_interpolated_wvlstep_0.06_spline_observed.csv'
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

out_dir = 'Ca_lines-red_2'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
os.chdir(out_dir)

tellurics_wvl_txt = 'tellurics_red_wvl.txt'
if not os.path.isfile(tellurics_wvl_txt):
    tellurics = list([])
    for field_id in selected_observation_fields:  # filter by date
        print 'Step1 - working on field '+str(field_id)
        fig, axes = plt.subplots(2, 1)
        spectra_row = np.where(field_id == observation_fields)
        # # rv values of every object in the field
        # rv_field = galah_param[spectra_row]['rv_guess']
        # rv_weight = np.abs(rv_field)
        # rv_weight = rv_weight/np.nanmax(rv_weight)
        #
        for row in spectra_row[0]:
            axes[0].plot(wvl_read, spectral_data[row, :], color='blue', alpha=0.05, linewidth=0.8)
        spectral_data_field = spectral_data[spectra_row[0], :]
        spectral_data_avg = np.nanpercentile(spectral_data_field, 90, axis=0)
        axes[0].plot(wvl_read, spectral_data_avg, color='black', linewidth=0.75)
        # sigma clipping with average data
        # TODO if needed
        # signal filtering
        spectral_data_avg_savgol = savgol_filter(spectral_data_avg, 7, 2)
        axes[0].plot(wvl_read, spectral_data_avg_savgol, color='red', linewidth=0.75)
        # find telluric abs lines
        idx_tellurics = argrelextrema(spectral_data_avg_savgol, np.less, order=3)
        tellurics.append(idx_tellurics)
        # add detected lines to the plot
        for abs_line in wvl_read[idx_tellurics]:
            axes[0].axvline(x=abs_line, color='black', linewidth=0.6)
        axes[0].set(xlim=(wvl_line_center - plot_range, wvl_line_center + plot_range), ylim=(0.4, 1.2), ylabel='Original spectra')
        #
        for row in spectra_row[0]:
            velocity_shift = galah_param[row]['rv_guess']
            if shift_for_barycentric:
                barycentric_vel = galah_barycentric[galah_barycentric['sobject_id']==galah_param[row]['sobject_id']]['vel_ccd3']
                velocity_shift -= barycentric_vel
            plt.plot(wvl_read/(1+velocity_shift*1000./C_LIGHT), spectral_data[row, :], color='blue', alpha=0.05, linewidth=0.8)
        for abs_line in line_list:
            axes[1].axvline(x=abs_line, color='black', linewidth=0.6)
        axes[1].set(xlim=(wvl_line_center - plot_range, wvl_line_center + plot_range), ylim=(0.4, 1.2), ylabel='RV shifted', xlabel='Wavelength')
        #
        plt.tight_layout()
        plt.savefig(str(field_id)+'_1.png', dpi=200)
        plt.close()
    # determine telluric lines from observed minima of functions
    counts, position = np.histogram(np.hstack(tellurics), bins=len(wvl_read), range=(0, len(wvl_read)))
    tellurics_selected = 100.*counts/get_fields > 40.
    for abs_line in wvl_read[tellurics_selected]:
        plt.axvline(x=abs_line, color='red', linewidth=0.6)
    plt.bar(wvl_read, counts, align='center', width=0.06, linewidth=0)
    plt.xlim = (wvl_line_center - plot_range, wvl_line_center + plot_range)
    plt.savefig('tellurics_hist.png', dpi=200)
    plt.close()
    # write results
    str_tellurics = ','.join([str(l) for l in wvl_read[tellurics_selected]])
    txt = open(tellurics_wvl_txt, 'w')
    txt.write(str_tellurics)
    txt.close()

tellurics_param_txt = 'tellurics_red_param.txt'
if not os.path.isfile(tellurics_param_txt):
    # function to be minimized
    def gaussian_fit(parameters, data, wvls, evaluate=True):
        n_keys = (len(parameters) - 1) / 3
        function_val = parameters['offset']*np.ones(len(wvls))
        for i_k in range(n_keys):
            function_val -= parameters['amp'+str(i_k)] * np.exp(-0.5 * (parameters['wvl'+str(i_k)] - wvls) ** 2 / parameters['std'+str(i_k)])
        if evaluate:
            likelihood = np.sum(np.power(data - function_val, 2))
            return likelihood
        else:
            return function_val

    telluric_lines = np.loadtxt(tellurics_wvl_txt, delimiter=',')
    for field_id in selected_observation_fields:  # filter by date
        print 'Step2 - working on field '+str(field_id)
        fig, axes = plt.subplots(2, 1)
        spectra_row = np.where(field_id == observation_fields)
        spectral_data_field = spectral_data[spectra_row[0], :]
        spectral_data_avg = np.nanpercentile(spectral_data_field, 90, axis=0)
        spectral_data_avg_savgol = savgol_filter(spectral_data_avg, 7, 2)
        axes[0].plot(wvl_read, spectral_data_avg_savgol, color='black', linewidth=0.5)
        # determine parameters for a comb of gaussian profiles that will be fitted to the spectra
        fit_param = Parameters()
        fit_param.add('offset', value=1., min=0.95, max=1.05)
        # for every line add std and amplitude of the absorption line
        fit_keys = list([])
        for i_l in range(len(telluric_lines)):
            key_std = 'std' + str(i_l)
            fit_param.add(key_std, value=0.01, min=0.0001, max=0.03)
            fit_keys.append(key_std)
            key_amp = 'amp' + str(i_l)
            fit_param.add(key_amp, value=0.1, min=0.0001, max=0.8)
            fit_keys.append(key_amp)
            key_wvl = 'wvl' + str(i_l)
            fit_param.add(key_wvl, value=telluric_lines[i_l], vary=False)
            fit_keys.append(key_wvl)
        # minimize the model
        fit_res = minimize(gaussian_fit, fit_param, method='brute',
                           args=(spectral_data_avg_savgol, wvl_read))
                           # **{'max_nfev': 20000, 'verbose': 1})
        # minim = Minimizer(gaussian_fit, fit_param, fcn_args=(spectral_data_avg_savgol, wvl_read))
        # fit_res = minim.emcee(steps=100, nwalkers=100, ntemps=1)
        fit_res.params.pretty_print()
        report_fit(fit_res)
        fitted_curve = gaussian_fit(fit_res.params, 0., wvl_read, evaluate=False)
        axes[0].plot(wvl_read, fitted_curve, color='red', linewidth=0.75)
        for abs_line in telluric_lines:
            axes[0].axvline(x=abs_line, color='blue', linewidth=0.6)
        axes[0].set(ylim=(0.4, 1.2), xlim=(wvl_line_center - plot_range, wvl_line_center + plot_range))
        plt.savefig(str(field_id) + '_2.png', dpi=200)
        plt.close()

