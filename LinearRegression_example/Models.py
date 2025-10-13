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
    
    def __init__(self, name, X, seed):
        '''
        Parameters
        ----------
        name (str): name of the model 
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        seed (int): seed for random number generator for initial values of weights and bias
        '''
        super().__init__(name) # inheriting all the Parent Class methods
        torch.manual_seed(seed) # setting the random number generator with the seed for reproducibility

        self.weights = torch.rand((X.shape[1],1),requires_grad=True) # initial values of weights w
        self.bias = torch.rand((1,1),requires_grad=True) # initial value of the coefficient b
        self.parameters = [self.weights,self.bias]
    
    def net_input(self,X):
        '''
        Method for calculating the input for the training or for a more complicated ML model (ANN)
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''

        return torch.matmul(X,self.weights) + self.bias
    
    def predict(self,X):
        '''
        Method for applying the ML model in a predictive way
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''
        
        prediction = self.net_input(X)
        return prediction
    


