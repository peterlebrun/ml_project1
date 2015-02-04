from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation    import CrossValidator, ModuleValidator
import common
import numpy as np

#data_set = common.get_bank_data_for_nn()
data_set = common.get_bc_data_for_nn()

test_means = []
test_std   = []
x_vals = [2, 3, 4, 5, 6, 7, 8]

for x in x_vals:
    means = []
    for i in range(20):
        trainer = BackpropTrainer(
                    buildNetwork(3, x, 1),
                    data_set,
                    verbose=True
        )

        print "%d %d" % (x, i)
        trainer.trainEpochs(3)
        cv = CrossValidator(trainer, trainer.ds, n_folds=5, valfunc=ModuleValidator.MSE)
        means.append(cv.validate())

    test_means.append(np.mean(means))
    test_std.append(np.std(means))

common.plot_nn_mse(
    title   = "Breast Cancer Survival Neural Network",
    x_label = "Number of hidden nodes",
    x_vals  = x_vals,
    y_means = np.array(test_means),
    y_std   = np.array(test_std)
)
