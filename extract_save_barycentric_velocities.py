from astropy.table import Table
from match_ids import *

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_objects = Table.read(galah_data_dir+'galah_objects.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')

out_path = galah_data_dir+'galah_barycentric.csv'
out_txt = open(out_path, 'w')
out_txt.close()
for sobj_id in galah_param['sobject_id']:
    print sobj_id
    comb_id = get_id(sobj_id)
    # print comb_id
    runs = match_cobid_with_runid(comb_id, galah_observations)
    # print runs
    galah_objects_use = galah_objects[np.logical_and(galah_objects['out_name'] >= np.int64(np.min(runs) * 1e6),
                                                     galah_objects['out_name'] <= np.int64((np.max(runs)+1) * 1e6))]
    bar_vel1 = get_barycentric(sobj_id, runs, galah_objects_use, ccd='1')
    bar_vel2 = get_barycentric(sobj_id, runs, galah_objects_use, ccd='2')
    bar_vel3 = get_barycentric(sobj_id, runs, galah_objects_use, ccd='3')
    bar_vel4 = get_barycentric(sobj_id, runs, galah_objects_use, ccd='4')
    out_txt = open(out_path, 'a')
    out_txt.write('{0},{1},{2},{3},{4}\n'.format(sobj_id, np.mean(bar_vel1), np.mean(bar_vel2), np.mean(bar_vel3), np.mean(bar_vel4)))
    out_txt.close()
    