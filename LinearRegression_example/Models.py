import pandas as pd
import numpy as np
import torch

class Models:
    '''
    Parent Class for different ML models. 
    It contains two methods ment to be overriden:
    net_input and predict
    '''

    def __init__(self,name):
        
        self.name = name
        print(f'Model choosen: {self.name}')

    def net_input(self):
        pass

    def predict(self):
        pass

class LinearRegression(Models):
    '''
    Child Class for a linear regression model 
    '''
    
    def __init__(self, name, w, b):
        '''
        Parameters
        ----------
        name (str): name of the model 
        w (torch tensor): dim = mx1 with m number of features. weights tensor
        b (torch tensor): dim = 1x1 is the bias tensor (intercept)
        '''
        super().__init__(name) # inheriting all the Parent Class methods
        self.w = w
        self.b = b
    
    def net_input(self,X):
        '''
        Method for calculating the input for the training or for a more complicated ML model (ANN)
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''

        return torch.matmul(X,self.w) + self.b
    
    def predict(self,X):
        '''
        Method for applying the ML model in a predictive way
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''
        
        prediction = self.net_input(X)
        return prediction
    
