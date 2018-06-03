import cPickle
import traj_dist.distance as  tdist
import numpy as  np
import multiprocessing

def trajectory_distance(traj_feature_map, traj_keys,  distance_type = "hausdorff", batch_size = 50, processors = 30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1],record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i!=0) & (i%batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         'geolife'))
            batch_number+=1
    pool.close()
    pool.join()


def trajecotry_distance_list(trajs, distance_type = "hausdorff", batch_size = 50, processors = 30, data_name = 'porto' ):
    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i!=0) & (i%batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         data_name))
            batch_number+=1
    pool.close()
    pool.join()

def trajectory_distance_batch(i, batch_trjs, trjs, metric_type = "hausdorff", data_name = 'porto'):
    if metric_type == 'lcss' or  metric_type == 'edr' :
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps= 0.003)
    # elif metric_type=='erp':
    #     trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
    cPickle.dump(trs_matrix, open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i), 'w'))
    print 'complete: '+str(i)


def trajectory_distance_combain(trajs_len, batch_size = 100, metric_type = "hausdorff", data_name = 'porto'):
    distance_list = []
    a = 0
    for i in range(1,trajs_len+1):
        if (i!=0) & (i%batch_size == 0):
            distance_list.append(cPickle.load(open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i))))
            print distance_list[-1].shape
    a = distance_list[-1].shape[1]
    distances = np.array(distance_list)
    print distances.shape
    all_dis = distances.reshape((trajs_len,a))
    print all_dis.shape
    cPickle.dump(all_dis,open('./features/'+data_name+'_'+metric_type+'_distance_all_'+str(trajs_len),'w'))
    return all_dis

if __name__ == '__main__':


    # Porto Datasets

    # traj_coord, lengths, max_len = cPickle.load(open('./data_taxi/porto_traj_coord_train'))
    # np_traj_coord = []
    # for t in traj_coord:
    #     np_traj_coord.append(np.array(t))
    # print np_traj_coord[0]
    # print np_traj_coord[1]
    #
    # distance_type = 'erp'
    # trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type)
    # trajectory_distance_combain(10000, batch_size=200, metric_type=distance_type)


    # Geolife Datasets
    # traj_coord, useful_grids, max_len = cPickle.load(open('./features/coordinate_trajectories', 'r'))
    # np_traj_coord = []
    # for t in traj_coord:
    #     np_traj_coord.append(np.array(t))
    # print np_traj_coord[0]
    # print np_traj_coord[1]
    # data_name = 'geolife'
    #
    # distance_type = 'lcss'
    # trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type, data_name=data_name)
    # trajectory_distance_combain(8200, batch_size=200, metric_type=distance_type, data_name=data_name)


    # Synthetic trajs
    traj_coord = cPickle.load(open('./data_taxi/synthetic_traj_coord', 'r'))
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print np_traj_coord[0]
    print np_traj_coord[1]
    data_name = 'synthetic'

    distance_type = 'discret_frechet'
    trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type,
                             data_name=data_name)
    trajectory_distance_combain(1800, batch_size=200, metric_type=distance_type, data_name=data_name)


    # distance_type = 'erp'
    # data_name = 'geolife'
    # traj_feature_map, rate, preiod = cPickle.load(open('./features/trajectories&frequentrate'))
    # print len(traj_feature_map.keys())
    # trajectory_distance(traj_feature_map, traj_feature_map.keys(), batch_size=200, processors=15,
    #                     distance_type=distance_type)
    # trajectory_distance_combain(8200, batch_size=200, metric_type=distance_type, data_name=data_name)