import random
import numpy as np
import cPickle


beijing_lat_range = [39.6,40.7]
beijing_lon_range = [115.9,117,1]

class Preprocesser(object):
    def __init__(self, delta = 0.005, lat_range = [1,2], lon_range = [1,2]):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x  = self._frange(dXMin,dXMax, self.delta)
        y  = self._frange(dYMin,dYMax, self.delta)
        self.x = x
        self.y = y

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def get_grid_index(self, tuple):
        test_tuple = tuple
        test_x,test_y = test_tuple[0],test_tuple[1]
        x_grid = int ((test_x-self.lon_range[0])/self.delta)
        y_grid = int ((test_y-self.lat_range[0])/self.delta)
        index = (y_grid)*(len(self.x)) + x_grid
        return x_grid,y_grid, index

    def traj2grid_seq(self, trajs = [], isCoordinate = False):
        grid_traj = []
        for r in trajs:
            x_grid, y_grid, index = self.get_grid_index((r[2],r[1]))
            # print (r[1]+tx,r[0]+ty)
            # print x_grid, y_grid, index
            grid_traj.append(index)
        privious = None
        hash_traj = []
        for index, i in enumerate(grid_traj):
            if privious==None:
                privious = i
                if isCoordinate == False:
                    hash_traj.append(i)
                elif isCoordinate == True:
                    hash_traj.append(trajs[index][1:])
            else:
                if i==privious:
                    pass
                else:
                    if isCoordinate == False:
                        hash_traj.append(i)
                    elif isCoordinate == True:
                        hash_traj.append(trajs[index][1:])
                    privious = i
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate =False):
        trajs_hash = []
        trajs_keys = traj_feature_map.keys()
        for traj_key in trajs_keys:
            traj = traj_feature_map[traj_key]
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))
        return trajs_hash

    def preprocess(self, traj_feature_map, isCoordinate = False):
        if not isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map)
            print 'gird trajectory nums {}'.format(len(traj_grids))

            useful_grids = {}
            count = 0
            max_len = 0
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len: max_len = len(traj)
                count += len(traj)
                for grid in traj:
                    if useful_grids.has_key(grid):
                        useful_grids[grid][1] += 1
                    else:
                        useful_grids[grid] = [len(useful_grids) + 1, 1]
            print len(useful_grids.keys())
            print count, max_len
            return traj_grids, useful_grids, max_len
        elif isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map, isCoordinate = isCoordinate)
            max_len = 0
            useful_grids = {}
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len: max_len = len(traj)
            return traj_grids, useful_grids, max_len



def trajectory_feature_generation(path ='./data/toy_trajs',
                                  lat_range = beijing_lat_range,
                                  lon_range = beijing_lon_range,
                                  min_length=50):
    fname = path.split('/')[-1].split('_')[0]
    trajs  =  cPickle.load(open(path))
    traj_index = {}
    max_len = 0
    preprocessor = Preprocesser(delta = 0.001, lat_range = lat_range, lon_range = lon_range)
    print  preprocessor.get_grid_index((lon_range[1],lat_range[1]))
    for i, traj in enumerate(trajs):
        new_traj = []
        coor_traj = []
        if (len(traj)>min_length):
            inrange = True
            for p in traj:
                lon, lat = p[0], p[1]
                if not ((lat > lat_range[0]) & (lat < lat_range[1]) & (lon > lon_range[0]) & (lon < lon_range[1])):
                    inrange = False
                new_traj.append([0, p[1], p[0]])

            if inrange:
                coor_traj = preprocessor.traj2grid_seq(new_traj, isCoordinate=True)
                # print coor_traj
                if len(coor_traj)==0:
                    print len(coor_traj)
                if ((len(coor_traj) >10) & (len(coor_traj)<150)):
                    if len(traj) > max_len: max_len = len(traj)
                    traj_index[i] = new_traj

        if i%200==0:
            print coor_traj
            print i, len(traj_index.keys())
    # print max_lat,max_lon,min_lat,min_lon
    print max_len
    print len(traj_index.keys())

    cPickle.dump(traj_index, open('./features/{}_traj_index'.format(fname),'w'))

    trajs, useful_grids, max_len = preprocessor.preprocess(traj_index, isCoordinate=True)

    print trajs[0]
    cPickle.dump((trajs,[],max_len), open('./features/{}_traj_coord'.format(fname), 'w'))
    # traj_grids = cPickle.load(open('./data_taxi/porto_traj_coord'))
    all_trajs_grids_xy = []
    min_x, min_y, max_x, max_y = 2000, 2000, 0, 0
    for i in trajs:
        for j in i:
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    print min_x, min_y, max_x, max_y

    for i in trajs:
        traj_grid_xy = []
        for j in i:
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            x = x - min_x
            y = y - min_y
            grids_xy = [y, x]
            traj_grid_xy.append(grids_xy)
        all_trajs_grids_xy.append(traj_grid_xy)
    print all_trajs_grids_xy[0]
    print len(all_trajs_grids_xy)
    print all_trajs_grids_xy[0]
    cPickle.dump((all_trajs_grids_xy,[],max_len), open('./features/{}_traj_grid'.format(fname), 'w'))

    return './features/{}_traj_coord'.format(fname), fname