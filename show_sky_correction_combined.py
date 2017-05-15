import os, imp
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
spectra_data_dir = '/media/storage/HERMES_REDUCED/dr5.1/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
# determine unique numbers of observation field
observation_fields = np.int64(galah_param['sobject_id']/1000.)
# np.unique(observation_fields)

field_id = 140315002501
os.chdir('Sky_spectrum')
for field_id in [140315002501]: #np.unique(observation_fields)[500:]:
    print field_id
    idx_read = spectra_row = np.where(field_id == observation_fields)
    for i_row in idx_read[0]:
        obj_data = galah_param[i_row]
        spectra_data = fits.open(spectra_data_dir+str(field_id)[:6]+'/combined/'+str(obj_data['sobject_id'])+'3.fits')
        sky_spectra = spectra_data[2].data - spectra_data[0].data
        wvl_start = spectra_data[0].header.get('CRVAL1')
        wvl_delta = spectra_data[0].header.get('CDELT1')
        spectra_data.close()
        wvl_values = wvl_start + np.arange(len(sky_spectra))*wvl_delta
        plt.plot(wvl_values, sky_spectra)
    plt.xlim((6490, 6520))
    plt.ylim((-200, 700))
    plt.savefig(str(field_id)+'.png', dpi=200)
    plt.close()
