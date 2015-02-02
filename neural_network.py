from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure           import TanhLayer, SoftmaxLayer
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import CrossValidator, ModuleValidator

from pandas import DataFrame, read_csv
from sklearn.cross_validation import train_test_split

import numpy as np

p = read_csv('data_banknote_authentication.txt',
             names=['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])

def normalize (vector):
    v_min = vector.min()
    v_range = vector.max() - vector.min()
    return (vector - v_min)/v_range

def get_pybrain_data_set(data, input_cols, target_cols = 1):
    data_set = SupervisedDataSet(input_cols, target_cols)
    for row in data:
        # get first X columns for inputs
        # (technically "all indices less than X")
        # get last column as target
        data_set.addSample(
            tuple(row[:input_cols]),
            tuple([row[input_cols]])
        )

    return data_set

# normalize all values
p['wti_var']  = normalize(p['wti_var'])
p['wti_skew'] = normalize(p['wti_skew'])
p['wti_curt'] = normalize(p['wti_curt'])
p['i_entr']   = normalize(p['i_entr'])

# shuffle up data
p.reindex(np.random.permutation(p.index))

trainer = BackpropTrainer(
            buildNetwork(4, 5, 1), # 2 input nodes, 3 hidden nodes, 1 output node
            get_pybrain_data_set(p.as_matrix(), 4),
            verbose = True
            )

#print(trainer.train())
#evaluator = ModuleValidator.classificationPerformance(trainer.module, trainer.ds)
cv = CrossValidator(trainer, trainer.ds, n_folds=5)
cv.setArgs(max_epochs = 2, verbose = True)
print(cv.validate())
