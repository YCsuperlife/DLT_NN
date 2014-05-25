import numpy as np

def nntest(nn,x,y):
	labels = nnpredict(nn,x)
	[dummy, expected] = numpy.amax(y,2)
	bad = np.where(labels!=expected[x])
	er = bad.size / x.shape[0]
	return [er, bad]
