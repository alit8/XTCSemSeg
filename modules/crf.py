import torch
import torch.nn as nn

from crfrnn.crf import CRF

class CRFRNN(nn.Module):
    def __init__(self, base_model, device):
        super(CRFRNN, self).__init__()
        self.base_model = base_model
        self.crf = CRF(n_ref=3, n_out=5, dev=device)

    def forward(self, x):
        input = x
        x = self.base_model(x)
        x = self.crf(x, input)
        return x