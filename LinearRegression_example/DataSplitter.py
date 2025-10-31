'''
DataSplitter.py is reliant on 3 files: 
- Data.csv - Provides the data to be split
- Input.yml - Provides the parameters for the split
- InputReader.py - Reads the Input.yml file
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from InputReader import InputReader
import torch

def get_random_state(input_file_name='Input.yml'):   
#Retrieves the random state from the input file.
   
    inp = InputReader(input_file_name)
    inp_read = inp.yml_reader()
    return inp_read.content['random_state']

def split_data(features, labels, test_size=0.2, random_state=None):
    '''
    Performs the train-test split on the provided features and labels.
    -----------------------------------------------------------------

    Parameters:
    Features (pd.DataFrame or np.array): the feature data.
    Labels (pd.DataFrame or np.array): the label data.
    test_size (float): proportion of the dataset to include in the test split.
    random_state (int): controls the shuffling applied to the data before applying the split.
    
    Returns:
    X_train, X_test, y_train, y_test (torch.Tensor): the split datasets as torch tensors.
    '''

    #Case 1: Full dataset requested (for cross-validation)
    if test_size == 0:
        #Return 100% of data as the training set, with an empty test set
        #we ensure the shapes match the expected output
        return features, None, labels, None

    else: 
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test

#should torrch sensor conversion be here or in the feature extraction?
