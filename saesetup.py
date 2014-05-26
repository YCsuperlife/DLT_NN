def saesetup(size):
	sae = {}
	for u in range(1, len(size)+1):
		sae[u] = nnsetup([size[u-1], size[u], size[u-1])
	return sae
