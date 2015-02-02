from pandas import read_csv
import numpy as np
from pybrain.datasets import SupervisedDataSet

def get_data(filename, column_names):
    p = read_csv(filename, names=column_names)

    return p.as_matrix()

def get_bank_data():
    return get_data('data_banknote_authentication.txt',
                    ['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])

def get_bank_data_inputs_and_target():
    data = get_bank_data()
    return p[:, :4], p[:, 4] #first four columns as input, final column as target

def get_bank_data_for_tree():
    inputs, target = get_bank_data_inputs_and_target()

    inputs[:, 0] = inputs[:, 0] <= 0 # if variance < 0
    inputs[:, 1] = inputs[:, 1] <= 0 # if skewness < 0
    inputs[:, 2] = inputs[:, 2] <= 0 # if curtosis < 0
    inputs[:, 3] = inputs[:, 3] <= 0 # if entropy  < 0

    return inputs, target

def normalize (vector):
    v_min = vector.min()
    v_range = vector.max() - vector.min()
    return (vector - v_min)/v_range

def get_bank_data_for_nn():
    data = get_bank_data()

    for i in range(3):
        data[:, i] = normalize(data[:, i])

    data_set = SupervisedDataSet(4, 1)

    for row in data:
        # get first X columns for inputs
        # (technically "all indices less than X")
        # get last column as target
        data_set.addSample(
            tuple(row[:4]),
            tuple([row[4]])
        )

    return data_set

def get_bc_data():
    data = get_data('breast_cancer.csv',
                    ['age', 'year', 'nodes_detected', 'survived_gt_5_yrs'])

    # data is initially classed as 1 = survival, 2 = death
    # adjust so that we have 0 = death, 1 = survival
    data[:, 3] = 2 - data[:, 3]

    return data

def get_bc_data_inputs_and_target():
    data = get_bc_data()
    return data[:, :3], data[:, 3] # first three columns as input, last column as target

# I know this is basically the same as "get_bank_data_for_nn"
# and I also know that I should "dry"
# and generally I would further refactor
# but it is late and I am tired
# and I just want to get this NN to work
# mea culpa
def get_bc_data_for_nn():
    data = get_bc_data()

    for i in range(3):
        data[:, i] = normalize(data[:, i])

    data_set = SupervisedDataSet(3, 1)

    for row in data:
        # get first X columns for inputs
        # (technically "all indices less than X")
        # get last column as target
        data_set.addSample(
            tuple(row[:3]),
            tuple([row[3]])
        )

    return data_set
