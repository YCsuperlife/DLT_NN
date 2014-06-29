import numpy as np
from struct import *
import nnopts
import saesetup as saes
import saetrain as saet

def MNISTexample(startN,howMany,bTrain=True,only01=False):
    if bTrain:
        fImages = open('train-images.idx3-ubyte','rb')
        fLabels = open('train-labels.idx1-ubyte','rb')
    else:
        fImages = open('t10k-images.idx3-ubyte','rb')
        fLabels = open('t10k-labels.idx1-ubyte','rb')

    # read the header information in the images file.
    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = unpack('>I',s1)[0]
    numIm = unpack('>I',s2)[0]
    rowsIm = unpack('>I',s3)[0]
    colsIm = unpack('>I',s4)[0]
    # seek to the image we want to start on
    fImages.seek(16+startN*rowsIm*colsIm)

    # read the header information in the labels file and seek to position
    # in the file for the image we want to start on.
    mnL = unpack('>I',fLabels.read(4))[0]
    numL = unpack('>I',fLabels.read(4))[0]
    fLabels.seek(8+startN)

    T = [] # list of (input, correct label) pairs
    
    for blah in range(0, howMany):
        # get the input from the image file
        x = []
        for i in range(0, rowsIm*colsIm):
            val = unpack('>B',fImages.read(1))[0]
            x.append(val/255.0)

        # get the correct label from the labels file.
        val = unpack('>B',fLabels.read(1))[0]
        y = []
        for i in range(0,10):
            if val==i: y.append(1)
            else: y.append(0)

        # if only01 is True, then only add this example if 0 or 1 is the
        # correct label.
        if not only01 or y[0]==1 or y[1]==1:
            T.append((x,y))
            
    fImages.close()
    fLabels.close()

    return T

train =  MNISTexample(0,2000)
for i in train:
	train_x = i[0]
	train_y = i[1]

test = MNISTexample(2000,2500)
for i in test:
	test_x = i[0]
	test_y = i[1]

newsae = saes.saesetupm(np.ndarray((784,100)))
newsae[1].activation_function = 'sigm'
newsae[1].learningRate = 1
newsae[1].inputZeroMaskedFraction = 0.5
opts = nnopts(100,1)

newsae = saet.saetrain(newsae, train_x, opts)
print newsae[1].W[1]





















