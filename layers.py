import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPLinearLayer(nn.Module):

  def __init__(self, cin, cout, epsilon=1e-6):
    super(SDPLinearLayer, self).__init__()

    self.weights = nn.Parameter(torch.empty(cout, cin))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.rand(cin))

    nn.init.xavier_normal_(self.weights)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)

    self.epsilon = epsilon

  def forward(self, x):
    q_ = self.q[:, None]
    q = torch.exp(q_)
    q_inv = torch.exp(-q_)
    T = 1/torch.abs(q_inv * self.weights.T @ self.weights * q).sum(1)
    y = torch.sqrt(T) * x
    out = F.linear(y, self.weights, self.bias)
    return out
