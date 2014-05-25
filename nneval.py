import numpy as np
import nnff

def nneval(nn, loss, train_x, train_y, val_x, val_y):
	nn.testing = 1
	nn = nnff(nn, train_x, train_y)
	loss['train_e'].append(nn.L)
	if val_x and val_y:
		nn = nnff(nn, val_x, val_y)
		loss['val_e'].append(nn.L)
	nn.testing = 0
	if nn.output == 'softmax':
		[er_train, dummy] = nntest(nn, train_x, train_y)
		loss['train_e_frac'].append(er_train)
		if val_x and val_y:
			[er_train, dummy] = nntest(nn, val_x, val_y)
			loss['val_e_frac'].append(er_val)
	return loss

	
