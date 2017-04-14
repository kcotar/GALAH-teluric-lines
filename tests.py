from astropy.table import Table
from match_ids import *

galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_objects = Table.read(galah_data_dir+'galah_objects.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')

sobj_id = galah_param[175104]['sobject_id']
# print sobj_id
comb_id = get_id(sobj_id)
# print comb_id
runs = match_cobid_with_runid(comb_id, galah_observations)
# print runs
bar_vel1 = get_barycentric(sobj_id, runs, galah_objects, ccd='1')
bar_vel2 = get_barycentric(sobj_id, runs, galah_objects, ccd='2')
bar_vel3 = get_barycentric(sobj_id, runs, galah_objects, ccd='3')
bar_vel4 = get_barycentric(sobj_id, runs, galah_objects, ccd='4')
print bar_vel1
print bar_vel2
print bar_vel3
print bar_vel4
print '---------------'
