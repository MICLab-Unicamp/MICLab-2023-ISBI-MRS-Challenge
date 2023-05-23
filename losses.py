import torch
import torch.nn as nn


class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss, self).__init__()

    # for the forward pass, a 1d ppm array must be passed and it's assumed that
    # it's valid for all sets
    def forward(self, x, y, ppm):
        # defining indexes of boundaries
        min_ind = torch.argmax(ppm[ppm <= 4])
        max_ind = torch.argmin(ppm[ppm >= 2.5])

        # selecting part of arrays pertaining to region of interest
        loss_x = x[:, min_ind:max_ind]
        loss_y = y[:, min_ind:max_ind]
        # calculate absolute loss mean value
        loss = torch.abs(loss_x - loss_y).mean(dim=1).mean(axis=0)
        return loss
