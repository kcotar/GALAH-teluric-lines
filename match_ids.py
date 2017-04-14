def match_cobid_with_runid(comb_id, observations):
    data_match = observations[observations['cob_id'] == comb_id]
    return list(set(data_match['run_id']))


def get_id(num):
    id_str = str(num)[:10]
    return int(id_str)


def get_barycentric(sobject_id, run_ids, objects, ccd='1'):
    # create observations ids from sobject ids and run ids
    obj_ids = list([])
    barycentric = list([])
    sobject_id_str = str(sobject_id)
    for run_id in run_ids:
        obj_str = str(run_id)+'00'+sobject_id_str[-3:]+ccd  # for ccd1, should be almost the same for every ccd
        obj_ids.append(int(obj_str))
    for obj in obj_ids:
        object_idx = objects['out_name'] == obj
        if object_idx.sum() == 1:
            barycentric.append(float(objects[object_idx]['barycentric']))
        else:
            print 'Problem.'
    print obj_ids
    return barycentric
