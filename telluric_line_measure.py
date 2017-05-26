import os, imp
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from lmfit import Parameters, minimize

imp.load_source('s_helper', '../Carbon-Spectra/helper_functions.py')
from s_helper import get_spectra_dr52


def fit_gauss_function(param, y_ref, wvl, func_value=False):
    param_values = param.valuesdict()
    y_fit = np.full_like(wvl, param_values['offset'])
    y_fit += param_values['amp']/(param_values['std'] * np.sqrt(2*np.pi)) * np.exp(-0.5 * (wvl - param_values['wvl_c']) ** 2 / param_values['std'] ** 2)
    if func_value:
        return y_fit
    else:
        return np.power(y_ref - y_fit, 2)

C_LIGHT = 299792458  # m/s

wvl_min = 6479
wvl_max = 6510

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
spectra_data_dir51 = '/media/storage/HERMES_REDUCED/dr5.1/'
spectra_data_dir52 = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv', format='ascii.csv')
# determine unique numbers of observation field

out_dir = 'Line_measurements'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)

wvl_investigate = 6498.7368 # Angstrom
out_txt = 'wvl_{0}_2.txt'.format(wvl_investigate)
txt_write = open(out_txt, 'w')
txt_write.close()
for i_obj in range(0, len(galah_param), 25):
    object_param = galah_param[i_obj]
    object_id = object_param['sobject_id']
    print object_id

    velocity_shift = -1.* object_param['v_bary']
    wvl_scale = (1 + velocity_shift * 1000. / C_LIGHT)

    object_spectra_ext0, wvl_ext0 = get_spectra_dr52(str(object_id), bands=[3], root=spectra_data_dir52, extension=0)
    object_spectra_ext2, _ = get_spectra_dr52(str(object_id), bands=[3], root=spectra_data_dir52, extension=2)
    sky_correction = object_spectra_ext0[0] - object_spectra_ext2[0]
    wvl_s = wvl_ext0[0] * wvl_scale

    est_offset = np.mean(sky_correction[np.abs(wvl_s - 6500.5) < 0.5])
    fit_param = Parameters()
    fit_param.add('offset', value=est_offset, vary=False)
    fit_param.add('amp', value=1)
    fit_param.add('wvl_c', value=wvl_investigate, vary=False)
    fit_param.add('std', value=0.1, min=0.01, max=0.3)
    fit_res = minimize(fit_gauss_function, fit_param,  # method='brute',
                       args=(sky_correction, wvl_s),
                       **{'nan_policy': 'omit'})
    fit_res_params = fit_res.params
    # fit_res_params.pretty_print()
    fit_func_res = fit_gauss_function(fit_res_params, 0, wvl_s, func_value=True)
    fit_res_params_values = fit_res_params.valuesdict()

    txt_write = open(out_txt, 'a')
    txt_write.write('{0},{1},{2}\n'.format(fit_res_params_values['wvl_c'],fit_res_params_values['amp'],fit_res_params_values['std']))
    txt_write.close()

    # plt.plot(wvl_s, sky_correction)
    # plt.plot(wvl_s, fit_func_res)
    # plt.axvline(x=wvl_investigate)
    # plt.xlim((wvl_min, wvl_max))
    # plt.savefig(str(object_id)+'.png', dpi=200)
    # plt.close()

