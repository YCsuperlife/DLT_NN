import numpy as np
def saesetupm(arrsize):
	sae = {}
	for u in range(1, np.size(arrsize)+1):
		sae[u] = nnsetup([size[u-1], size[u], size[u-1]])
	return sae
