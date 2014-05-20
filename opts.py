class opts(object):
	def __init__(self,validation,batchsize,numepochs,plot):
                self.validation = validation
                self.batchsize = batchsize
                self.numepochs = numepochs
                self.plot = plot
        def __str__(self):
                return str(self.output)
        def __repr__(self):
                return str(self)
        def __getitem__(self,y):
                return self.y
        def __setitem__(self,y,z):
                self.y = z

