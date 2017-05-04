from astropy.table import Table
from match_ids import *

galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_objects = Table.read(galah_data_dir+'galah_objects.fits')
galah_objects_type = Table.read(galah_data_dir+'galah_objects-type.fits')
galah_observations = Table.read(galah_data_dir+'galah_observations.fits')
galah_objects = galah_objects[(galah_objects['out_name'] % 10) == 3]
print galah_objects[:10]

print 'Start'
idx_rand = np.int64(np.random.uniform(0, len(galah_param), 200))
for i in range(1):
    sobj_id = galah_param[i]['sobject_id']
    sobj_id = 170206004201253
    print sobj_id
    comb_id = get_id(sobj_id)
    # print comb_id
    runs = match_cobid_with_runid(comb_id, galah_observations)
    print runs
    # print runs
    # galah_objects_use = galah_objects[np.logical_and(galah_objects['out_name'] >= np.int64(np.min(runs) * 1e6),
    #                                                  galah_objects['out_name'] <= np.int64((np.max(runs)+1) * 1e6))]
    # bar_vel3 = get_barycentric(sobj_id, runs, galah_objects_use, ccd='3')
    galah_objects_type_use = galah_objects_type[np.logical_and(galah_objects_type['out_name'] >= np.int64(np.min(runs) * 1e6),
                                                               galah_objects_type['out_name'] <= np.int64((np.max(runs) + 1) * 1e6))]
    sky_sobjects = get_skyfibers_field(runs[0], galah_objects_type_use, ccd=3)
    print sky_sobjects
    print len(sky_sobjects)
