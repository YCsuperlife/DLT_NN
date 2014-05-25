from numpy.random import random as nprandom
import numpy as np
import math

class nn(object):
	def __init__(self,size):
		self.size = size
		self.activationfn = 'tanh'
		self.learningRate = 2
		self.momentum = 0.5
		self.scaling_learningRate = 1
		self.weightPenaltyL2 = 0
		self.nonSparsityPenalty = 0
		self.sparsityTarget = 0.05
		self.inputZeroMaskedFraction = 0
		self.dropoutFraction = 0
		self.testing = 0
		self.output = 'sigm'
		self.n = len(size)
		self.W = {}
		self.vW = {}
		self.dW = {}
		self.p = {}
		self.a = {}
		self.e = {}
		self.L = {}
		for i in range(1,self.n):
			self.W[i] = (nprandom((self.size[i],self.size[i-1]+1)) - 0.5) *2 * 4 * math.sqrt(6/(self.size[i] + self.size[i-1])) 
			self.vW[i] = np.zeros((self.W[i].shape))
			self.p[i+1] = np.zeros((1, self.size[i+1]))
	def __str__(self):
                return str(self.output)
        def __repr__(self):
                return str(self)
        def __getitem__(self,y):
                return self.y
        def __setitem__(self,y,z):
                self.y = z

def nnsetup(architecture):
	#architecture is a 1xN vector of layer sizes
	new_net = nn(architecture)
	return new_net

#new_net = nnsetup([4,5,6])
#print new_net.W
#print new_net.vW
#print new_net.p

