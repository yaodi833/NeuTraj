# Data path
corrdatapath = './features/toy_traj_coord'
gridxypath = './features/toy_traj_grid'
distancepath = './features/toy_discret_frechet_distance_all_1800'

# Training Prarmeters
GPU = "4"
learning_rate = 0.01
seeds_radio=0.2
epochs = 100000
batch_size = 20
sampling_num = 10

distance_type = distancepath.split('/')[2].split('_')[1]
data_type = distancepath.split('/')[2].split('_')[0]

if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8




# Test Config
datalength = 1800
em_batch = 900
test_num = 100

# Model Parameters
d = 128
stard_unit = False
incell = True
recurrent_unit = 'LSTM' #LSTM or SimpleRNN
spatial_width  = 2

gird_size = [1100, 1100]


def config_to_str():
   configs = 'learning_rate = {} '.format(learning_rate)+ '\n'+\
             'mail_pre_degree = {} '.format(mail_pre_degree)+ '\n'+\
             'seeds_radio = {} '.format(seeds_radio) + '\n' +\
             'epochs = {} '.format(epochs)+ '\n'+\
             'datapath = {} '.format(corrdatapath) +'\n'+ \
             'datatype = {} '.format(data_type) + '\n' + \
             'corrdatapath = {} '.format(corrdatapath)+ '\n'+ \
             'distancepath = {} '.format(distancepath) + '\n' + \
             'distance_type = {}'.format(distance_type) + '\n' + \
             'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
             'batch_size = {} '.format(batch_size)+ '\n'+\
             'sampling_num = {} '.format(sampling_num)+ '\n'+\
             'incell = {}'.format(incell)+ '\n'+ \
             'stard_unit = {}'.format(stard_unit)
   return configs


if __name__ == '__main__':
    print '../model/model_training_600_{}_acc_{}'.format((0),1)
    print config_to_str()