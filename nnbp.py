import numpy as np

def nnbp(nn):
	n = nn.n
	sparsityError = 0
	d = {}
	if nn.output == 'sigm':
		d[n] = np.multiply(-nn.e,np.nultiply(nn.a[n],(1-nn.a[n])))
	elif np.output == 'softmax' or np.output == 'linear':
		d[n] = -nn.e
	for i in range(n-1, 1, -1):
		if nn.activation_function == 'sigm':
			d_act = np.multiply(nn.a[i],(1-nn.a[i]))
		elif nn.activation_function == 'tanh':
			d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * np.power(nn.a[i],2))
		if nn.nonSparsityPenalty > 0:
			pi = np.tile(nn.p[i], (nn.a[i].shape[0],1))
			sparsityError = [np.zeros(nn.a[i].shape[0],1), nn.nonSparsityPenalty * np.divide(-nn.sparsityTarget,np.divide(pi + (1 - nn.sparsityTarget),(1 - pi)))]
		if i+1==n:
			d[i] = np.multiply(d[i+1] * nn.W[i] + sparsityError, d_act)
		else:
			d[i] = np.multiply((d[i + 1][:,1:d[i+1].shape[0]] * nn.W[i] + sparsityError),d_act)
		if nn.dropoutFraction > 0:
			d[i] = np.multiply(d[i],np.ones(d[i].shape[0],1).append(nn.dropOutMask[i]))

	for i in range(1,n):
		if i+1==n:
			nn.dw[i] = (d[i+1] * nn.a[i])/d[i+1].shape[0]
		else:
			nn.dW[i] = (d[i+1][:,1:d[i+1].shape[0]] * nn.a[i])/d[i+1].shape[0] 

	return nn

