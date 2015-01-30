#from pandas import DataFrame, read_csv
#import matplotlib.pyplot as plt
#import pandas as pd
#import sys
#import numpy as np
#from sklearn.cross_validation import train_test_split
#from sklearn import tree
#import numpy as np
#
##print 'Python version: ' + sys.version
##print 'Pandas version: ' + pd.__version__
#
#p = read_csv('data_banknote_authentication.txt',
#             names=['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])
#
## plt.plot(p['wti_var']) - split on var > 0, var < 0
## plt.plot(p['wti_skew']) - split on var > 0, var < 0
## plt.plot(p['wti_curt']) - split on 0, 5
## plt.plot(p['i_entr']) - split on 0
#
##### DECISION TREE
#
#p['var_lte_0']  = p['wti_var'] <= 0
#p['skew_lte_0'] = p['wti_skew'] <= 0
#p['curt_lte_0'] = p['wti_curt'] <= 0
#p['entr_lte_0'] = p['i_entr'] <= 0
#
#train, test = train_test_split(p, test_size = 0.33)
#
#x_val = train[:,5:]
#y_val = train[:,4]
#
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x_val, y_val)
#print(clf)
#print(clf.predict(test[:,5:]).sum())
#
#from sklearn.externals.six import StringIO
#import pydot
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf('out.pdf')

# TREES
#http://scikit-learn.org/stable/modules/tree.html
#http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html
#import numpy as np
#import matplotlib.pyplot as plt
#
#from sklearn import datasets
#from sklearn.tree import DecisionTreeClassifier
#
##parameters
#n_classes=3
#plot_colors = "bry"
#plot_step = 0.02
#
#iris = datasets.load_iris()
#
#for pairidx, pair in enumerate([[0,1], [0,2], [0,3],
#                                [1,2], [1,3], [2,3]]):
## We only take the two corresponding features
#    X = iris.data[:, pair]
#    y = iris.target
#
#    # Shuffle
#    idx = np.arange(X.shape[0])
#    np.random.seed(13)
#    np.random.shuffle(idx)
#    X = X[idx]
#    y = y[idx]
#
#    # Train
#    clf = DecisionTreeClassifier().fit(X, y)
#
#    #plot the decision boundary
#    plt.subplot(2, 3, pairidx + 1)
#
#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                         np.arange(y_min, y_max, plot_step))
#
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#    plt.xlabel(iris.feature_names[pair[0]])
#    plt.ylabel(iris.feature_names[pair[1]])
#    plt.axis("tight")
#
#    # plot the training points
#    for i, color in zip(range(n_classes), plot_colors):
#        idx = np.where(y == i)
#        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                    cmap=plt.cm.Paired)
#
#    plt.axis("tight")
#
#plt.suptitle("Decision surface of a decision tree using pair features")
#plt.legend()
#plt.show()




#### KNN
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
#http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
#http://scikit-learn.org/stable/modules/neighbors.html
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

#### SVM
#http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
#http://scikit-learn.org/stable/modules/svm.html
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
##we creat an instance of the SVM and fit out data.  We do not scale our data since we want to plot the support vectors
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
#

## Boosting
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# Author: noel dawe (noel.dawe@gmail.com

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples = 200,
                                 n_features = 2,
                                 n_classes = 2,
                                 random_state = 1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples = 300,
                                 n_features = 2,
                                 n_classes = 2,
                                 random_state = 1)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

# create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), algorithm="SAMME", n_estimators=200)

bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"


plt.figure(figsize=(10, 5))

#plot the decision boundaries

plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))


Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

#plot the training points

for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c = c, cmap=plt.cm.Paired,
                label="Class %s" % n)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc="upper right")
plt.xlabel("Decision boundary")

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins = 10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha = .5)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc="upper right")
plt.ylabel("Samples")
plt.xlabel("Decision Scores")

plt.subplots_adjust(wspace=0.25)
plt.show()
