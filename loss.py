import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from kornia.filters import sobel


__all__ = ["L1Loss2d", "MSELoss2d", "CrossEntropyLoss2d", "CrossEntropyLoss2dLabelSmooth", "SmoothEdgeLoss"]

class L1Loss2d(_WeightedLoss):

    def __init__(self, reduction='mean'):
        super(L1Loss2d, self).__init__()

        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        # output = torch.argmax(output, axis=1).float()
        # target = target.float()

        output = output.squeeze(1)

        return self.loss(output, target)

class MSELoss2d(_WeightedLoss):

    def __init__(self, reduction='mean'):
        super(MSELoss2d, self).__init__()

        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        output = torch.argmax(output, axis=1).float()
        target = target.float()
        return self.loss(output, target)

class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, ignore_label=-100, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """

        return self.nll_loss(output, target)

class ConsistencyLoss2d(_WeightedLoss):

    def __init__(self, weight=None, ignore_label=-100, reduction='mean', xtasks=['depth']):
        super(ConsistencyLoss2d, self).__init__()
        self.xtasks = xtasks
        self.cardinal = len(self.xtasks)
        self.depth_loss = L1Loss2d()
        self.normal_loss = L1Loss2d()
        self.nll_loss = FocalLoss2d(weight=weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target, xoutputs, xtargets):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """

        depth_loss = 0
        normal_loss = 0

        if 'depth' in self.xtasks:
          depth_loss = self.depth_loss(xoutputs['depth'], xtargets['depth'])

        if 'normal' in self.xtasks:
          normal_loss = self.normal_loss(xoutputs['normal'], xtargets['normal'])

        seg_loss = self.nll_loss(output, target)

        print(seg_loss, depth_loss, normal_loss)

        return self.cardinal * seg_loss + depth_loss / 500 + normal_loss / 50


class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, ignore_label=-100, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        n_classes = output.size(1)
        # batchsize, num_class = input.size()
        # log_probs = F.log_softmax(inputs, dim=1)
        # targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

        return self.nll_loss(output, targets)

class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1,2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt) ** self.gamma) * self.alpha * logpt
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class SmoothEdgeLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=-100, reduction='mean', l1_factor=0.2, l2_factor=0.0, edge_factor=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.edge_factor = edge_factor
        self.fl = FocalLoss2d(alpha=self.alpha, gamma=self.gamma, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, output, target, data):

        def layout_gradient(output, σ=5.0):
            return 1 - torch.exp(-sobel(output.unsqueeze(1).float()) / σ)

        # if output.dim()>2:
        #     output = output.contiguous().view(output.size(0), output.size(1), -1)
        #     output = output.transpose(1,2)
        #     output = output.contiguous().view(-1, output.size(2)).squeeze()
        # if target.dim()==4:
        #     target = target.contiguous().view(target.size(0), target.size(1), -1)
        #     target = target.transpose(1,2)
        #     target = target.contiguous().view(-1, target.size(2)).squeeze()
        # elif target.dim()==3:
        #     target = target.view(-1)
        # else:
        #     target = target.view(-1, 1)

        loss = 0
        ''' per-pixel classification loss '''
        seg_loss = F.nll_loss(F.log_softmax(output, dim=1), target, ignore_index=255)
        loss += seg_loss
        # terms['loss/cla'] = seg_loss

        ''' area smoothness loss '''
        if self.l1_factor or self.l2_factor:
            l_loss = F.mse_loss if self.l2_factor else F.l1_loss
            l1_λ = self.l1_factor or self.l2_factor
            # TODO ignore 255
            onehot_target = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
            l1_loss = l_loss(output, onehot_target)
            loss += l1_loss * l1_λ
            # terms['loss/area'] = l1_loss

        ''' layout edge constraint loss '''
        if self.edge_factor:
            _, prediction = torch.max(output, 1)
            edge_map = layout_gradient(prediction).squeeze(1)
            target_edge = data['edge'].to(device=edge_map.device)
            edge_loss = F.binary_cross_entropy(edge_map.double(), target_edge.double())
            loss += edge_loss * self.edge_factor
            # terms['loss/edge'] = edge_loss

        return loss