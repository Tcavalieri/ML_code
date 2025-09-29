import pandas as pd
import numpy as np
#import sklearn

# The code in this class has benn written using most of the code
# presented in chapter 2 od the book 'Machine Learning with PyTorch and Scikit-Learn'
# from the recommended resources for this project.

class LinearRegressionGD:
    '''
    class for the implementation and training of a linear regression gradient descent model. 
    '''
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        
        self.eta = eta # learning rate
        self.n_iter = n_iter # number of iterations
        self.random_state = random_state # seed for number generator

    def fit(self, X, y):
        '''
        X: input  np array of dimension nxm
        n=number of data points
        m=number of features per point
        '''
        rgen = np.random.RandomState(self.random_state) # random seed generator
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # initial values of weights w
        self.b_ = np.array([0.]) # initial value of the coefficient b
        self.losses_ = [] # losses array initialisation

        # training routine
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        '''
        basic function that calculates the output of the linear regression 
        '''
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        '''
        function for prediction
        '''
        return self.net_input(X)