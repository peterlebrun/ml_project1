from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from sklearn.cross_validation import train_test_split
import numpy as np

print 'Python version: ' + sys.version
print 'Pandas version: ' + pd.__version__

p = read_csv('data_banknote_authentication.txt',
             names=['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])

# plt.plot(p['wti_var']) - split on var > 0, var < 0
# plt.plot(p['wti_skew']) - split on var > 0, var < 0
# plt.plot(p['wti_curt']) - split on 0, 5
# plt.plot(p['i_entr']) - split on 0

p['var_lte_0'] = p['wti_var'] <= 0
p

#p_train, p_test = train_test_split(p, test_size = 0.33)
#
#arr = p_train[:,3]
#
##plt.plot(arr)
#
#from sklearn import tree
#x_val = p_train[:,:3]
#y_val = p_train[:,4]
#
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x_val, y_val)
#clf.predict(p_test[:,:3]).sum()
