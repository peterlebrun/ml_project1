from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from sklearn import tree, cross_validation
from sklearn.externals.six import StringIO
import pydot
import common

#inputs, target = common.get_bank_data_for_tree()
inputs, target = common.get_bc_data_for_tree()

# create tree
x_vals = [1, 2, 3]
test_means = []
test_std   = []

for i in x_vals:
    clf = tree.DecisionTreeClassifier(max_depth = i)
    scores = cross_validation.cross_val_score(clf, inputs, target, cv = 25)
    test_means.append(scores.mean())
    test_std.append(scores.std())

common.plot_vals(
    title = "Breast Cancer Survival Decision Tree",
    x_label = "Max Tree Depth",
    x_vals  = x_vals,
    y_means = np.array(test_means),
    y_std   = np.array(test_std)
)

#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf('bank_notes.pdf')
