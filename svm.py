#### KNN
##http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
##http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
##http://scikit-learn.org/stable/modules/neighbors.html
#
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#from sklearn import neighbors, datasets
#
#n_neighbors = 15
#
#iris = datasets.load_iris()
#X = iris.data[:, :2]
#y = iris.target
#
#h = .02 # step size
#
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
#for weights in ['uniform', 'distance']:
#    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#    clf.fit(X, y)
#
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

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

from pandas import DataFrame, read_csv
import numpy as np
import common

#inputs, target = common.get_bank_data_inputs_and_target()
inputs, target = common.get_bc_data_inputs_and_target()

### SVM
for C in [0.5, 1.0, 1.5]:
    print("%0.2f" % (C))
    svc      = svm.SVC(kernel='linear', C=C)
    rbf_svc  = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
    lin_svc  = svm.LinearSVC(C=C)
    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        scores = cross_validation.cross_val_score(clf, inputs, target, cv = 5)
        print("%s: Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean(), scores.std() * 2))

#### SVM
##http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
##http://scikit-learn.org/stable/modules/svm.html
#
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import svm, datasets
#
#iris = datasets.load_iris()
#X = iris.data[:, :2] # only grab the first two features
#y = iris.target
#
#h = .02 # step size in the mesh
#
###we creat an instance of the SVM and fit our data.  We do not scale our data since we want to plot the support vectors
#C = 1.0 # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)
#
## create a mesh to plot in
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#
## title for the plots
#titles = ['CSV with linear kernel',
#          'LinearSVC (linear kernel)',
#          'SVC with RBF kernel',
#          'SVC with polynomial (degree 3) kernel']
#
#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#    # Plot the decision boundary.
#    # For that, we will assign a color to each
#    # point in the mesh (x_min, m_max]x[y_min, y_max]
#    plt.subplot(2, 2, i + 1)
#    plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#    print(np.c_[xx.ravel(), yy.ravel()])
#    exit()
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#    # put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    plt.contour(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
#    # plot also the training points
#    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#    plt.xlabel('Sepal length')
#    plt.ylabel('Sepal width')
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(titles[i])
#
#plt.show()
