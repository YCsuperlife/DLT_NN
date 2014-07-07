import numpy as np
import nntrain as nnt

def saetrain(sae, x, opts):
	for i in range(1,len(sae)+1):
		print 'Training AE ' + str(i) + '/' + str(len(sae)+1)
		sae[i] = nnt.nntrain(sae[i],x ,x, opts)
		t = nnff(sae[i],x,x)
		x = t.a[2]
		x = x[:,1:x.shape[0]]
	return sae
 
