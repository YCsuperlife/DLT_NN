import numpy as np

def nnpredict(nn ,x):
	nn.testing =1
	nn = nnff(nn,x,np.zeros(x.shape[0], nn.size[-1]))
	nn.testing = 0
	[dummy, i] = np.amax[nn.a[max(nn.a.keys())],2]
	labels = i
	return labels
