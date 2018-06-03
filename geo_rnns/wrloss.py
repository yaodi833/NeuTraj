import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch
from tools import config
import numpy as np
class WeightMSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            self.weight.append(0.)
            for traj_index in range(sampling_num):
                self.weight.append(np.array([config.sampling_num - traj_index]))

        self.weight = np.array(self.weight)
        sum = np.sum(self.weight)
        self.weight = self.weight /sum
        self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad = False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU = False):
        div = target - input.view(-1,1)
        if isReLU:
            div = F.relu(div.view(-1,1))
        square = torch.mul(div.view(-1,1), div.view(-1,1))
        weight_square = torch.mul(square.view(-1,1), self.weight.view(-1,1))

        loss = torch.sum(weight_square)
        return loss

class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False)

        negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), True)

        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss,negative_mse_loss])
        return loss