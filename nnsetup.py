def nnsetup(architecture):
	#architecture is a 1xN vector of layer sizes
	



class nn(object):
	def __init__(self,size):
		self.size = size
		self.activationfn = 0
		self.learningRate = 0
		self.momentum = 0
		self.scaling_learningRate = 0
		self.weightPenaltyL2 = 0
		self.nonSparsityPenalty = 0
		self.sparsityTarget = 0
		self.inputZeroMaskedFraction = 0
		self.dropoutFraction = 0
		self.testing = 0
		self.output = 0
		self.n = len(size)
		self.W = []
	def __str__(self):
                return str(self.output)
        def __repr__(self):
                return str(self)
        def __getitem__(self,y):
                return self.y
        def __setitem__(self,y,z):
                self.y = z
	def initweights():
		for i in range(2:self.n):
			self.W.append(
