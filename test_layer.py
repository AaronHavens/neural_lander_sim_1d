from layers import SDPLinearLayer
import numpy as np
import torch
from time import time
from tqdm import tqdm

n_in = 3
n_out = 3

F = SDPLinearLayer(n_in, n_out)

N = 10000
max_lip = 0.0
bound = 1
for j in tqdm(range(N)):
	x = np.random.uniform(-bound, bound, size=(n_in,))
	y = np.random.uniform(-bound, bound, size=(n_in,))

	Fx = F(torch.from_numpy(x).float()).detach().numpy()
	Fy = F(torch.from_numpy(y).float()).detach().numpy()

	norm_in = np.linalg.norm(x-y, 2)
	norm_out = np.linalg.norm(Fx-Fy, 2)

	Lip = norm_out/norm_in

	if Lip > max_lip: max_lip = Lip

print('max lipschitz constant: ', max_lip)