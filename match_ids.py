import numpy as np


def match_cobid_with_runid(comb_id, observations):
    data_match = observations[observations['cob_id'] == comb_id]
    return list(set(data_match['run_id']))


def get_id(num):
    id_str = str(num)[:10]
    return int(id_str)


def get_barycentric(sobject_id, run_ids, objects, ccd=1):
    # create observations ids from sobject ids and run ids
    barycentric = list([])
    sobject_id_str = str(sobject_id)
    for run_id in run_ids:
        obj_str = str(run_id)+'00'+sobject_id_str[-3:] + str(ccd)  # for ccd1, should be the same for every ccd (but it is not)
        object_idx = np.where(objects['out_name'] == int(obj_str))
        if len(object_idx[0]) == 1:
            barycentric.append(float(objects[object_idx]['barycentric']))
        else:
            print 'Problem.'
    return barycentric


def get_object_type(sobject_id, run_ids, objects, ccd=1):
    type = list([])
    sobject_id_str = str(sobject_id)
    for run_id in run_ids:
        obj_str = str(run_id) + '00' + sobject_id_str[-3:] + str(ccd)  # ccd number is irrelevant in this case
        object_idx = np.where(objects['out_name'] == int(obj_str))
        if len(object_idx[0]) == 1:
            type.append(float(objects[object_idx]['type']))
        else:
            print 'Problem.'
    return type


def get_skyfibers_field(run_id, objects, ccd=1):
    objects_runid = np.int64(objects['out_name']/1e6)
    idx_run = np.where(objects_runid == run_id)
    # object subset
    objects_sub = objects[idx_run]
    idx_sky = np.where(np.logical_and(objects_sub['type'] == 'S',
                                      objects_sub['out_name'] % 10 == ccd))
    return objects_sub[idx_sky]['out_name'].data


def combined_to_individual(s_id, run_id):
    s_id_str = str(s_id)
    run_id = str(run_id)
    out = run_id+'00'+s_id_str[-3:]
    return np.int64(out)
