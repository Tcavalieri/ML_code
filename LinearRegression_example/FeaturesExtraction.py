#import sklearn
import pandas as pd
import numpy as np
import torch

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

        dim = [self.data,len(names)] # extracting dimension of the pointsxfeatures tensor
        labels = torch.empty((dim[0],1),requires_grad=False) # initialise the labels tensor 
        labels[:,0] = torch.tensor(data_df[label_name].values)
        
        features = torch.empty((dim[0],dim[1]),requires_grad=False) # initialise the pointsxfeatures tensor 
        for i in range(dim[1]):
            data_df[names[i]] = data_df[names[i]].apply(lambda x: float(x))
            features[:,i] = torch.tensor(data_df[names[i]].values).double() # .double method used for compatibility of dtype
        
        return labels, features


        

