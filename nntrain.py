import numpy as np

def nntrain(nn, train_x, train_y, opts, val_x, val_y):
	assert type(train_x) == float
	loss = {}
	loss['train_e'] = []
	loss['train_e_frac'] = []
	loss['val_e'] = []
	loss['val_e_frac'] = []
	if val_x:
		opts.validation = 1
	else:
		opts.validation = 0
	'''fhandle = []
	if opts.plot ==1 :
		fhandle = figure();
	'''
	m = train_x.shape[0]
	batchsize = opts.batchsize
	numepochs = opts.numepochs
	numbatches = m/batchsize
	assert m % batchsize == 0	
	L = np.zeros(numepochs*numbatches,1)
	n = 1
	for i in range(0,numepochs):
		kk = np.random.permutation(range(1,m+1))
		for j in range(0, numbatches):
			batch_x = train_x[(kk[(j-1)*batchsize+1:j*batchsize]),:]
			batch_y = train_y[(kk[(j-1)*batchsize+1:j*batchsize]),:]
			nn = nnff(nn, batch_x, batch_y)
			nn = nnbp(nn)
			nn = nnapplygrads(nn)
			L[n] = nn.L
			n = n+1
		if opts.validation ==1:
			loss = nneval(nn, loss,train_x, val_x, val_y)
			print '; Full-batch train mse = %f, val mse = %f' % (loss.train.e(end), loss.val.e(end))
		else:
			loss = nneval(nn, loss, train_x, train_y)
			print '; Full-batch train err = %f' % loss.train.e(end)
	return nn, L
