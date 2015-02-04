import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

from pandas import DataFrame, read_csv
import numpy as np
import common

inputs, target = common.get_bank_data_inputs_and_target()
#inputs, target = common.get_bc_data_inputs_and_target()

### SVM
train_vals = [0.001, 0.01, 0.1, 1, 10]
test_means = []
test_std = []

for C in [1]:
    print(C)
    #clf      = svm.SVC(kernel='linear', C=C)
    #clf  = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    clf = svm.SVC(kernel='poly', degree=3, C=C)

    scores = cross_validation.cross_val_score(clf, inputs, target, cv = 10)
    test_means.append(scores.mean())
    test_std.append(scores.std())

common.plot_vals(
    title   = "Breast Cancer Survival SVM: Polynomial Kernel (Deg 3)",
    x_label = "Regularization Parameter",
    x_vals  = train_vals,
    y_means = np.array(test_means),
    y_std   = np.array(test_std)
)
