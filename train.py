from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os

if __name__ == '__main__':
    print 'os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"])
    print config.config_to_str()
    trajrnn = NeuTrajTrainer(tagset_size = config.d, batch_size = config.batch_size,
                             sampling_num = config.sampling_num)
    trajrnn.data_prepare(griddatapath = config.gridxypath, coordatapath = config.corrdatapath,
                         distancepath = config.distancepath, train_radio = config.seeds_radio)
    load_model_name = None
    trajrnn.neutraj_train(load_model = None, in_cell_update=config.incell,
                          stard_LSTM=config.stard_unit)
