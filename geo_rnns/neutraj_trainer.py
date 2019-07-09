import tools.test_methods as tm
import time, os, cPickle
import numpy as np
import torch

from tools import  config
from tools import sampling_methods as sm
from neutraj_model import NeuTraj_Network
from wrloss import WeightedRankingLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]  = config.GPU

def pad_sequence(traj_grids, maxlen=100, pad_value = 0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0])*pad_value
        while (len(traj) < maxlen):
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs


class NeuTrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate = config.learning_rate):

        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self, griddatapath = config.gridxypath,
                     coordatapath = config.corrdatapath,
                     distancepath = config.distancepath,
                     train_radio = config.seeds_radio):
        dataset_length = config.datalength
        traj_grids, useful_grids, max_len = cPickle.load(open(griddatapath, 'r'))
        self.trajs_length = [len(j) for j in traj_grids][:dataset_length]
        self.grid_size = config.gird_size
        self.max_length = max_len
        grid_trajs = [[[i[0]+config.spatial_width , i[1]+config.spatial_width] for i in tg]
                      for tg in traj_grids[:dataset_length]]

        traj_grids, useful_grids, max_len = cPickle.load(open(coordatapath, 'r'))
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy] for r in t] for t in traj_grids]

        coor_trajs = traj_grids[:dataset_length]
        train_size = int(len(grid_trajs)*train_radio/self.batch_size)*self.batch_size
        print train_size

        grid_train_seqs, grid_test_seqs = grid_trajs[:train_size], grid_trajs[train_size:]
        coor_train_seqs, coor_test_seqs = coor_trajs[:train_size], coor_trajs[train_size:]

        self.grid_trajs = grid_trajs
        self.grid_train_seqs = grid_train_seqs
        self.coor_trajs = coor_trajs
        self.coor_train_seqs = coor_train_seqs
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append([coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)



        print "Padded Trajs shape"
        print len(pad_trjs)
        self.train_seqs = pad_trjs[:train_size]
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen= max_len))
        distance = cPickle.load(open(distancepath,'r'))
        max_dis = distance.max()
        print 'max value in distance matrix :{}'.format(max_dis)
        print config.distance_type
        if config.distance_type == 'dtw':
            distance = distance/max_dis
        print "Distance shape"
        print distance[:train_size].shape
        train_distance = distance[:train_size, :train_size]

        print "Train Distance shape"
        print train_distance.shape
        self.distance = distance
        self.train_distance = train_distance

    def batch_generator(self, train_seqs, train_distance):
        j = 0
        while j< len(train_seqs):
            anchor_input, trajs_input, negative_input,distance,negative_distance = [],[],[],[],[]
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            for i in range(self.batch_size):
                # sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                sampling_index_list = sm.distance_sampling(self.distance,len(self.train_seqs), j + i)
                negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i)

                trajs_input.append(train_seqs[j+i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                if not batch_trajs_keys.has_key(j+i):
                    batch_trajs_keys[j+i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j+i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))

                    if not batch_trajs_keys.has_key(traj_index):
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            #normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input), np.array(batch_trajs_input)],
                   [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
                   [np.array(distance),np.array(negative_distance)])
            j = j + self.batch_size


    def trained_model_eval(self, print_batch = 10 ,print_test = 100,save_model = True, load_model = None,
                           in_cell_update = True, stard_LSTM = False):

        spatial_net = NeuTraj_Network(4, self.target_size, self.grid_size,
                                      self.batch_size, self.sampling_num,
                                      stard_LSTM= stard_LSTM, incell= in_cell_update)

        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)

            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            print 'len(embeddings): {}'.format(len(embeddings))
            print embeddings.shape
            print embeddings[0].shape

            acc1 = tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
                                 similarity=True, print_batch=print_test, r10in50=True)
            return acc1


    def neutraj_train(self, print_batch = 10, print_test = 100, save_model = True, load_model = None,
                      in_cell_update= True, stard_LSTM = False):

        spatial_net = NeuTraj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM= stard_LSTM, incell= in_cell_update)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()


        if load_model != None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            print 'len(embeddings): {}'.format(len(embeddings))
            print embeddings.shape
            print embeddings[0].shape

            tm.test_model(self,embeddings, test_range=range(len(self.train_seqs), len(self.train_seqs)+config.test_num),
                                 similarity=True, print_batch=print_test, r10in50=True)
        for epoch in range(config.epochs):
            spatial_net.train()
            print "Start training Epochs : {}".format(epoch)
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()
            for i, batch in enumerate(self.batch_generator(self.train_seqs, self.train_distance)):

                inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]

                trajs_loss, negative_loss = spatial_net(inputs_arrays,inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                loss = mse_loss_m(trajs_loss,positive_distance_target,negative_loss,negative_distance_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                optim_time = time.time()
                if not in_cell_update:
                    spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()
                if (i + 1) % print_batch == 0:
                    print 'Epoch [{}/{}], Step [{}/{}], Positive_Loss: {}, Negative_Loss: {}, Total_Loss: {}, ' \
                          'Update_Time_cost: {}, All_Time_cost: {}'.\
                           format(epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                              mse_loss_m.trajs_mse_loss.item(), mse_loss_m.negative_mse_loss.item(),
                              loss.item(), batch_end-optim_time,batch_end-start)

            end = time.time()
            print 'Epoch [{}/{}], Step [{}/{}], Positive_Loss: {}, Negative_Loss: {}, Total_Loss: {}, ' \
                  'Time_cost: {}'. \
                format(epoch + 1, config.epochs,  i + 1 , len(self.train_seqs) // self.batch_size,
                       mse_loss_m.trajs_mse_loss.item(), mse_loss_m.negative_mse_loss.item(),
                       loss.item(), end - start)


            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            print 'len(embeddings): {}'.format(len(embeddings))
            print embeddings.shape
            print embeddings[0].shape


            acc1 = tm.test_model(self,embeddings, test_range=range(int(len(self.padded_trajs)*0.8),
                                                                   int(len(self.padded_trajs)*0.8)+config.test_num),
                                 similarity=True, print_batch=print_test)

            print acc1
            if save_model:
                save_model_name = './model/{}_{}_{}_training_{}_{}_incell{}'\
                                      .format(config.data_type, config.distance_type, config.recurrent_unit,
                                              len(embeddings), str(epoch), in_cell_update) +\
                                '_config_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.stard_unit, config.learning_rate,
                                                                         config.batch_size, config.sampling_num,
                                                                         config.seeds_radio, config.data_type,
                                                                         str(stard_LSTM), config.d) +\
                                '_train_{}_test_{}_{}_{}_{}_{}.h5'.format(acc1[0], acc1[1], acc1[2], acc1[3],
                                                                          acc1[4], acc1[5])
                print save_model_name
                torch.save(spatial_net.state_dict(), save_model_name)
