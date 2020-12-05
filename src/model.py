import torch
import torch.nn as nn
import numpy as np


class GCNModel(nn.Module):
    def __init__(self, dropout=0.1):
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        pass

    def encode(self, inputs):
        return

    def forward(self, x):
        pass


class GraphConv(nn.Module):
    def __init__(self, dropout=0.1):
        pass

    def forward(self):
        pass
