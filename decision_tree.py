from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from sklearn import tree, cross_validation
#from sklearn.externals import StringIO
import pydot
import common

inputs, target = common.get_bank_data_for_tree()

# create tree
clf = tree.DecisionTreeClassifier()
scores = cross_validation.cross_val_score(clf, inputs, target, cv = 5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
exit()

#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf('out.pdf')

#parameters
n_classes=3
plot_colors = "bry"
plot_step = 0.02

iris = datasets.load_iris()

for pairidx, pair in enumerate([[0,1], [0,2], [0,3],
                                [1,2], [1,3], [2,3]]):
# We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    #plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    # plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using pair features")
plt.legend()
plt.show()
