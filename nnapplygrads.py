import numpy as np

def nnapplygrads(nn):
	for i in range(1, nn.n):
		if nn.weightPenaltyL2 >0:
			dW = nn.dW[i] + nn.weightPenaltyL2 * np.zeros(nn.W[i].shape[0],1).append(nn.W[i][:,1:nn.W[i].shape[0]])
		else:
			dW = nn.dW[i]
		dW = nn.learningRate * dW
		if nn.momentum > 0:
			nn.vW[i] = nn.momentum*nn.vW[i] + dW
			dW = nn.vW[i]
		nn.W[i] = nn.W[i] - dW
	return nn
