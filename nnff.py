import numpy as np

def nnff(nn,x,y):
	n = nn.n
	m = x.shape[0]
	x = np.ones(m,1)
	x.append([1])
	nn.a[1] = x
	for i in range(2,n):
		if nn.activationfn == 'sigm':
			nn.a[i] = sigmoid(nn.a[i-1] * nn.W[i-1])
		elif nn.activationfn == 'tahn_opt':
			nn.a[i] = np.tanh(nn.a[i-1] * nn.W[i-1])
		if nn.nonSparsitypenalty > 0:
			nn.p[i] = 0.99 * nn.p[i] + 0.01 * np.mean(nn.a[i], axis=1)
		nn.a[i] = np.ones(m,1).append(nn.a[i])
	if nn.output == 'sigm':
		nn.a[n] = sigmoid(nn.a[n - 1] * nn.W[n - 1])
	elif nn.output == 'linear':
		nn.a[n] = nn.a[n - 1] * nn.W[n - 1]
	elif nn.output == 'softmax':
		nn.a[n] = nn.a[n - 1] * nn.W[n - 1]
		nn.a[n] = np.exp(np.subtract(nn.a[n], max(nn.a[n],[],2)))
		nn.a[n] = np.divide(nn.a[n], np.sum(nn.a[n],2))
	
	nn.e = y - nn.a[n]
	if nn.output == 'softmax':
		nn.L = -np.sum(np.sum(np.multiply(y,np.log(nn.a[n])))) / m
	elif (nn.output == 'sigm' or nn.output == 'linear'):
		nn.L = 1/2 * np.sum(np.sum(np.power(nn.e,2))) / m	
	return nn	
