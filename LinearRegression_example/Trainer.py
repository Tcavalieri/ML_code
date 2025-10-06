import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim

class Trainer:
    '''
    Class for training different ML models utilising different optimisation algorithms
    '''

    def __init__(self,model_obj,eta,n_iter,random_state):
        '''
        parameters
        ----------
        model_obj (obj): instance of Models class containing the ML model selected
        eta (float): learning rate, value in between 0 and 1
        n_iter (int): number of iterations
        random_state (int): seed for random number generator
        '''

        self.model_obj = model_obj
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def sgd_optim(self,X,y):
        '''
        Method that train the ML model using stochastich gradient descent as implemented in PyTorch
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        y (torch tensor): dim = nx1 with n number of labeled points
        '''
        
        torch.manual_seed(self.random_state) # setting the random number generator with the seed for reproducibility

        self.model_obj.w = torch.rand((X.shape[1],1),requires_grad=True) # initial values of weights w
        self.model_obj.b = torch.rand((1,1),requires_grad=True) # initial value of the coefficient b
        self.losses = [] # losses array initialisation
        criterion = nn.MSELoss() # loss function Mean Square Error
        self.optimizer = optim.SGD([self.model_obj.w,self.model_obj.b],self.eta) # optimiser object

        # training routine
        for i in range(self.n_iter):

            output = self.model_obj.net_input(X) 
            loss = criterion(output,y)
            self.optimizer.zero_grad() # zeroing gradients
            loss.backward() # calculate gradients
            self.optimizer.step() # updating parameters
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

        torch.manual_seed(self.random_state) # setting the random number generator with the seed for reproducibility
        
        self.model_obj.w = torch.rand((X.shape[1],1),requires_grad=False) # initial values of weights w
        self.model_obj.b = torch.rand((1,1),requires_grad=False) # initial value of the coefficient b
        self.losses = [] # losses array initialisation

        # training routine
        for i in range(self.n_iter):
            output = self.model_obj.net_input(X)
            errors = (y - output)
            self.model_obj.w += self.eta * 2.0 * (torch.matmul(torch.t(X),errors)) / X.shape[0]
            self.model_obj.b += self.eta * 2.0 * torch.mean(errors)
            errors_2 = torch.pow(errors,2)
            loss = torch.mean(errors_2)
            self.losses.append(loss)
        return self