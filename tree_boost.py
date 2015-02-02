import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, ensemble, cross_validation
import common

inputs, target = common.get_bank_data_inputs_and_target()

inputs[:, 0] = inputs[:, 0] <= 0 # if variance < 0
inputs[:, 1] = inputs[:, 1] <= 0 # if skewness < 0
inputs[:, 2] = inputs[:, 2] <= 0 # if curtosis < 0
inputs[:, 3] = inputs[:, 3] <= 0 # if entropy  < 0

clf = ensemble.AdaBoostClassifier(
        tree.DecisionTreeClassifier(max_depth = 3),
        algorithm="SAMME",
        n_estimators=200
)
scores = cross_validation.cross_val_score(clf, inputs, target, cv = 5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
exit()

#
#plot_colors = "br"
#plot_step = 0.02
#class_names = "AB"
#
#
#plt.figure(figsize=(10, 5))
#
##plot the decision boundaries
#
#plt.subplot(121)
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                     np.arange(y_min, y_max, plot_step))
#
#
#Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#cs = plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
#plt.axis("tight")
#
##plot the training points
#
#for i, n, c in zip(range(2), class_names, plot_colors):
#    idx = np.where(y == i)
#    plt.scatter(X[idx, 0], X[idx, 1],
#                c = c, cmap=plt.cm.Paired,
#                label="Class %s" % n)
#
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#plt.legend(loc="upper right")
#plt.xlabel("Decision boundary")
#
## Plot the two-class decision scores
#twoclass_output = bdt.decision_function(X)
#plot_range = (twoclass_output.min(), twoclass_output.max())
#plt.subplot(122)
#for i, n, c in zip(range(2), class_names, plot_colors):
#    plt.hist(twoclass_output[y == i],
#             bins = 10,
#             range=plot_range,
#             facecolor=c,
#             label='Class %s' % n,
#             alpha = .5)
#x1, x2, y1, y2 = plt.axis()
#plt.axis((x1, x2, y1, y2 * 1.2))
#plt.legend(loc="upper right")
#plt.ylabel("Samples")
#plt.xlabel("Decision Scores")
#
#plt.subplots_adjust(wspace=0.25)
#plt.show()
