import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from match_ids import *
from astropy.table import Table

imp.load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
from helper_functions import get_skyspectra_dr52

spectra_dir = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_objects_type = Table.read(galah_data_dir+'galah_objects-type.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')
os.chdir('Sky_spectrum_from_fits')
for comb_id in list([1402090017,1402100017,1402110027,1402120011,1612110026,1612110031,1702060042,1702060047,1702060052]):
    # comb_id = 1702060042
    print comb_id
    runs = match_cobid_with_runid(comb_id, galah_observations)
    print runs
    galah_objects_type_use = galah_objects_type[np.logical_and(galah_objects_type['out_name'] >= np.int64(np.min(runs) * 1e6),
                                                               galah_objects_type['out_name'] <= np.int64((np.max(runs) + 1) * 1e6))]
    for run in runs:
        for ccd in list([1,2,3,4]):
            sky_sobjects = get_skyfibers_field(run, galah_objects_type_use, ccd=ccd)
            print sky_sobjects
            for sky_sobject in sky_sobjects:
                sky_sobject_str = str(sky_sobject)[:-1]
                print 'Reading '+sky_sobject_str
                spectra, wavelengths = get_skyspectra_dr52(sky_sobject_str, bands=[ccd], root=spectra_dir)
                if spectra is None:
                    continue
                plt.plot(wavelengths[0], spectra[0], linewidth=0.2)
            print 'Save graph'
            plt.xlim((np.min(wavelengths), np.max(wavelengths)))
            plt.ylim((-50, 150))
            plt.savefig('{0}_{1}.png'.format(ccd, run), dpi=450)
            # plt.show()
            plt.close()

