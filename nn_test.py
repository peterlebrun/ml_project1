from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet

net = buildNetwork(2, 3, 1) # 2 input nodes, 3 hidden nodes, 1 output node
net.activate([2, 1]) # no clue what this means

#net['in'] #see input layer
#net['hidden0'] #see hidden layer
#net['out'] #see output layer

# Change up hidden layer (which defaults to sigmoid squashing function)
net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer)

# Change up output layer (which defaults to a linear layer)
net = buildNetwork(2, 3, 2, hiddenClass=TanhLayer, outclass=SoftmaxLayer)

# use bias
net = buildNetwork(2, 3, 1, bias=True)

# construct data set

ds = supervisedDataSet(2, 1) #two input parameters and one output parameter

# train for XOR function
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

len(ds) # outputs 4
for input, target in ds:
    print input, target

ds['input']
ds['target']
