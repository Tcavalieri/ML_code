#import sklearn
import pandas as pd
import numpy as np

class FeaturesExtraction:
    '''
    Class f
    or reading and extracting features from raw datas. The methods
    of this class only read *.csv (comma-separated Values) files for now.
    '''

    def __init__(self,name):
        self.name = name # name of the file
        self.data = 0 # initialisation of the number of total points
        self.tot_missing = 0
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
        optional function to analyse the data
        '''
        raw_data = self.data_reading()
        cols = raw_data.columns
        print(cols)
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
        Auxiliary function to manipulate some data
        '''
        data_new = self.data_miss()
        data_new[col] = data_new[col].apply(lambda x: x/1000)
        
        return data_new

    def feature_extraction(self,label_name,names):
        '''
        function to extract features and labels from the data.
        variables:
        label_name (str): name of the label column in the df
        names (list): list of strings  for the features 
        '''
        data_df = self.data_miss()
        #if check == True:
        #    data_df = self.data_preproc(col)
        
        labels = np.array(data_df[label_name])
        dim = [self.data,len(names)]
        features = np.empty((dim[0],dim[1]))
        for i in range(len(names)):
            features[:,i] = data_df[names[i]]
        
        return labels, features


        
