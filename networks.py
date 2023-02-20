import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SDPLinearLayer


class SLLNetwork(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = SDPLinearLayer(12, 25)
        self.fc2 = SDPLinearLayer(25, 30)
        self.fc3 = SDPLinearLayer(30, 15)
        self.fc4 = SDPLinearLayer(15, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


