from tools import preprocess
from tools.distance_compution import trajectory_distance_combain,trajecotry_distance_list
import cPickle
import numpy as  np


def distance_comp(coor_path):
    traj_coord = cPickle.load(open(coor_path, 'r'))[0]
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print np_traj_coord[0]
    print np_traj_coord[1]
    print len(np_traj_coord)

    distance_type = 'discret_frechet'

    trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type,
                             data_name=data_name)

    trajectory_distance_combain(1800, batch_size=200, metric_type=distance_type, data_name=data_name)

if __name__ == '__main__':
    coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/toy_trajs')
    distance_comp(coor_path)
