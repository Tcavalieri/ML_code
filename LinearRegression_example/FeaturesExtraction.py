import sklearn
import pandas as pd
import numpy as np
import torch
from DataSplitter import split_data #import the new splitting function
from sklearn.model_selection import KFold 

class FeaturesExtraction:
    '''
    Class for reading and extracting features from raw data. The methods
    of this class only read *.csv (comma-separated Values) files for now.
    '''

    def __init__(self,name):
        self.name = name # name of the file
        self.data = 0 # initialisation of the number of total points
        self.tot_missing = 0 # initialisation for the number of missing values
        self.uniques = {}
        self.missing = {}
    
    def data_reading(self):
        '''
        function for reading the raw data file and creating a data frame
        '''
        df = pd.read_csv(self.name) 
        return df
    
    def data_analysis(self):
        '''
        Function to analyse the data and storng important information in the instance (self.data)
        '''
        raw_data = self.data_reading()
        cols = raw_data.columns
        miss = 0
        for col in cols:
            self.uniques[col] = len(raw_data[col].unique())
            self.missing[col] = len(raw_data.loc[raw_data[col]=='-'])
            miss = miss + self.missing[col]
        self.data = len(raw_data)
        self.tot_missing = miss
    
    def data_miss(self):
        '''
        function to replace missing values
        '''
        data = self.data_reading()
        data = data.replace('-',0.0)
        return data

    def data_preproc(self,col):
        '''
        Auxiliary function to manipulate some data (optional and not used for now)
        '''
        data_new = self.data_miss()
        data_new[col] = data_new[col].apply(lambda x: x/1000)
        
        return data_new

    def feature_extraction(self,label_name,names):
        '''
        function to extract features and labels from the data.
        Parameters
        ----------
        label_name (str): name of the label column in the dataframe
        names (list): list of strings  for the features 
        Return
        ------
        labels (torch tensor): dim = nx1 with n number of labled points
        features (torch tensor): dim = nxm with n number of points and m number of features
        '''
        data_df = self.data_miss() # replace missing values

        labels = data_df[label_name].values.astype(float).reshape(-1,1) # extraction of labels
        features = data_df[names].values # extraction of features
        for i, name in enumerate(names):
            features[:,i] = data_df[name].values

        features = features.astype(float)
        
        return labels, features
    
    def data_split(self,features,labels,test_ratio,random_state):
        
        '''
        function to split the data into training and testing sets.
        Parameters
        ----------
        label_name (np.array or str): name of the label column in the dataframe
        names (list): list of strings  for the features 
        test_size (float): proportion of the dataset to include in the test split.
        random_state (int): controls the shuffling applied to the data before applying the split.
        Return
        ------
        X_train, X_test, y_train, y_test (torch.Tensor): the split datasets as torch tensors.
        '''

        #Perform the split using the new split_data function from DataSplitter.py
        X_train_np, X_test_np, y_train_np, y_test_np = split_data(
            features, labels, test_size=test_ratio, random_state=random_state
        )   

        #Convert NumPy arrays to PyTorch tensors
        # 1. Training Features (always present)
        X_train = torch.tensor(X_train_np, dtype=torch.double, requires_grad=False)
        
        # 2. Testing Features (check if None - occurs when test_ratio=0.0)
        if X_test_np is None:
            # Create an empty tensor if no test data exists
            X_test = torch.empty((0, features.shape[1]), dtype=torch.double, requires_grad=False)
        else:
            X_test = torch.tensor(X_test_np, dtype=torch.double, requires_grad=False)
        
        # 3. Training Labels (always present)
        y_train = torch.tensor(y_train_np, dtype=torch.double, requires_grad=False)
        
        # 4. Testing Labels (check if None - occurs when test_ratio=0.0)
        if y_test_np is None:
            # Create an empty tensor if no test data exists
            y_test = torch.empty((0, labels.shape[1]), dtype=torch.double, requires_grad=False)
        else:
            y_test = torch.tensor(y_test_np, dtype=torch.double, requires_grad=False)
        
        return X_train, X_test, y_train, y_test
    
