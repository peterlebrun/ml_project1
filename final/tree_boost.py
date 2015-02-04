import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, ensemble, cross_validation
import common

#inputs, target = common.get_bank_data_for_tree()
inputs, target = common.get_bc_data_for_tree()

x_vals = np.linspace(0.25, 1.75, 7)
test_means = []
test_std   = []

for x in x_vals:
    print(x)
    clf = ensemble.AdaBoostClassifier(
            tree.DecisionTreeClassifier(max_depth = 3),
            algorithm="SAMME",
            learning_rate = x,
            n_estimators = 200
    )
    scores = cross_validation.cross_val_score(clf, inputs, target, cv = 20)
    test_means.append(scores.mean())
    test_std.append(scores.std())

common.plot_vals(
    title = "Breast Cancer Survival with AdaBoost",
    x_label = "Learning Rate",
    x_vals = x_vals,
    y_means = np.array(test_means),
    y_std = np.array(test_std)
)
