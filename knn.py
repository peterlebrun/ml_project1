import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, cross_validation

from pandas import DataFrame, read_csv
import numpy as np
import common

#inputs, target = common.get_bank_data_inputs_and_target()
inputs, target = common.get_bc_data_inputs_and_target()

# the prevailing wisdom seems to be to set "k" to the sqrt of the number of instances
k = np.floor(np.sqrt(len(inputs)))

for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(k, weights = weights)
    scores = cross_validation.cross_val_score(clf, inputs, target, cv = 10)
    print("%s: Accuracy: %0.2f (+/- %0.2f)" % (weights, scores.mean(), scores.std() * 2))

#iris = datasets.load_iris()
#X = iris.data[:, :2]
#y = iris.target

#h = .02 # step size

#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##Plot the decision boundary.  For that we will assign a color
##to each point in the mesh [x_min, x_max] x [y_min, y_max]
#
#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
## put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    plt.figure()
#    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
##plot also the training points
#    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
#
#plt.show()
