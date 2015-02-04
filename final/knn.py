import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, cross_validation
from pandas import DataFrame, read_csv

import common

inputs, target = common.get_bank_data_inputs_and_target()
#inputs, target = common.get_bc_data_inputs_and_target()

ks = np.linspace(1, 30, 10)
for weights in ['uniform', 'distance']:
    test_means = []
    test_std   = []

    for k in ks:
        clf = neighbors.KNeighborsClassifier(k, weights = weights)
        scores = cross_validation.cross_val_score(clf, inputs, target, cv = 20)
        test_means.append(scores.mean())
        test_std.append(scores.std())

    common.plot_vals(
        title = "Counterfeit Bank Note k means clustering: " + weights + " weighting",
        x_label = "K",
        x_vals = ks,
        y_means = np.array(test_means),
        y_std = np.array(test_std)
    )
