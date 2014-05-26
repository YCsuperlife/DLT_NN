import numpy as np

def saetrain(sae, x, opts):
	for i in range(1,len(sae)+1):
		print 'Training AE ' + i + '/' + len(sae)+1
		sae[i] = nntrain(sae[i],x ,x, opts)
		t = nnff(sae[i],x,x)
		x = t.a[2]
		x = x[:,1:x.shape[0]]
	return sae
 
