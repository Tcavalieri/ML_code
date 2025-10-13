import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim

class Trainer:
    '''
    Class for training different ML models utilising different optimisation algorithms
    '''

    def __init__(self,name,model_obj,eta,n_iter):
        '''
        parameters
        ----------
        name (str): name of the optimiser used
        model_obj (obj): instance of Models class containing the ML model selected
        eta (float): learning rate, value in between 0 and 1
        n_iter (int): number of iterations
        '''
        self.name = name
        self.model_obj = model_obj
        self.eta = eta
        self.n_iter = n_iter

    def optim_selection(self):
        '''
        Methods to choose which optimisation algorithm to use for the training
        '''
        if self.name == 'SGD':
            
            self.optimiser = optim.SGD(self.model_obj.parameters,self.eta)
            print('The optimiser is Stochastic Gradient Descent')

        elif self.name == 'Adam':

            self.optimiser = optim.Adam(self.model_obj.parameters,self.eta)
            print('The optimiser is Adam')
            
        return self

    def training(self,X,y):
        '''
        Method that train the ML model using the selected optimiser as implemented in PyTorch
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        y (torch tensor): dim = nx1 with n number of labeled points
        '''
        
        self.losses = [] # losses array initialisation
        criterion = nn.MSELoss() # loss function Mean Square Error

        self.optim_selection()
        # training routine
        for i in range(self.n_iter):

            output = self.model_obj.net_input(X) 
            loss = criterion(output,y)
            self.optimiser.zero_grad() # zeroing gradients
            loss.backward() # calculate gradients
            self.optimiser.step() # updating parameters
            self.losses.append(loss.item())
        return self
    
    def gd_optim(self,X,y):
        '''
        Method that train the ML model using gradient descent as implemented in PyTorch
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        y (torch tensor): dim = nx1 with n number of labeled points
        '''
        
        self.model_obj.weights.requires_grad_(False)
        self.model_obj.bias.requires_grad_(False)
        self.losses = [] # losses array initialisation

        # training routine
        for i in range(self.n_iter):
            output = self.model_obj.net_input(X)
            errors = (y - output)
            self.model_obj.weights += self.eta * 2.0 * (torch.matmul(torch.t(X),errors)) / X.shape[0]
            self.model_obj.bias += self.eta * 2.0 * torch.mean(errors)
            errors_2 = torch.pow(errors,2)
            loss = torch.mean(errors_2)
            self.losses.append(loss)

        return self

