from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure           import TanhLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import CrossValidator, ModuleValidator
import common

## bank data
#trainer = BackpropTrainer(
#            buildNetwork(4, 6, 1), # input nodes, hidden nodes, output nodes
#            common.get_bank_data_for_nn(),
#            verbose = True
#            )
#
#cv = CrossValidator(trainer, trainer.ds, n_folds=5)
#cv.setArgs(max_epochs = 2, verbose = True)
#print(cv.validate())

# bc data
trainer = BackpropTrainer(
            buildNetwork(3, 4, 1), # input nodes, hidden nodes, output nodes
            common.get_bc_data_for_nn(),
            verbose = True
            )

cv = CrossValidator(trainer, trainer.ds, n_folds=12)
cv.setArgs(max_epochs = 2, verbose = True)
print(cv.validate())
