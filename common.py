from pandas import DataFrame, read_csv
import numpy as np

def get_bank_data():
    p = read_csv('data_banknote_authentication.txt',
                 names=['wti_var', 'wti_skew', 'wti_curt', 'i_entr', 'cl'])
    p = p.as_matrix()

    inputs = p[:, :4] #the entropy doesn't make a difference
    target = p[:, 4]

    return inputs, target
