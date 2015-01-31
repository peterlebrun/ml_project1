from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet

from pandas import DataFrame, read_csv

p = read_csv('data_banknote_authentication.txt',
             names=['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])

def normalize (vector):
    v_min = vector.min()
    v_range = vector.max() - vector.min()
    return (vector - v_min)/v_range

# plt.plot(p['wti_var']) - split on var > 0, var < 0
# plt.plot(p['wti_skew']) - split on var > 0, var < 0
# plt.plot(p['wti_curt']) - split on 0, 5
# plt.plot(p['i_entr']) - split on 0

# convert into 0-1 values

p['wti_var']  = normalize(p['wti_var'])
p['wti_skew'] = normalize(p['wti_skew'])
p['wti_curt'] = normalize(p['wti_curt'])
p['i_entr']   = normalize(p['i_entr'])

ds = SupervisedDataSet(4, 1)

for row in p:

    input  = tuple(p[:3])
    target = tuple(p['cl'])
    ds.addSample(input, target)

#target = tuple(p[:, 4])

#ds.addSample(input, target)

#net = buildNetwork(2, 3, 1) # 2 input nodes, 3 hidden nodes, 1 output node
#net.activate([2, 1]) # no clue what this means
#
##net['in'] #see input layer
##net['hidden0'] #see hidden layer
##net['out'] #see output layer
#
## Change up hidden layer (which defaults to sigmoid squashing function)
#net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer)
#
## Change up output layer (which defaults to a linear layer)
#net = buildNetwork(2, 3, 2, hiddenClass=TanhLayer, outclass=SoftmaxLayer)
#
## use bias
#net = buildNetwork(2, 3, 1, bias=True)
#
## construct data set
#
#ds = supervisedDataSet(2, 1) #two input parameters and one output parameter
#
## train for XOR function
#ds.addSample((0, 0), (0,))
#ds.addSample((0, 1), (1,))
#ds.addSample((1, 0), (1,))
#ds.addSample((1, 1), (0,))
#
#len(ds) # outputs 4
#for input, target in ds:
#    print input, target
#
#ds['input']
#ds['target']

