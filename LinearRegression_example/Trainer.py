import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold # <-- NEW IMPORT
from Models import LinearRegression

class Trainer:
    '''
    Class for training different ML models utilising different optimisation algorithms
    '''

    def __init__(self,name,model_obj,eta,n_iter,batch_size):
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
        self.batch_size = batch_size

     #--------------------------------------------------------------------------------------------------------
    #Cross-Validation Method taken from Gemini; how can i work on this and adapt it myself??
    #--------------------------------------------------------------------------------------------------------

    def cross_validate(self, X, y, n_splits=5): ###should this split be done outside the trainer class and instead in the input file??
        '''
        Method that performs K-Fold Cross-Validation
        
        **Parameters**
        ----------
        X (torch tensor): All features (full dataset).
        y (torch tensor): All labels (full dataset).
        n_splits (int): Number of folds for CV (default is 5).
        
        Returns
        ------
        avg_loss (float): The average loss (MSE) across all folds.
        all_fold_losses (list): List of loss values for each fold.
        '''
        #KFold ensures deterministic (reproducible) splits across CV runs/folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=234)
        all_fold_losses = []
        criterion = nn.MSELoss()

        #Convert PyTorch tensors to NumPy arrays for KFold to work easily
        X_np = X.detach().numpy()
        y_np = y.detach().numpy()
        
        print(f'Starting {n_splits}-Fold Cross-Validation...')
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_np)):
            print(f'--- Fold {fold+1}/{n_splits} ---')
            
            # 1. Split data for the current fold
            X_train_fold = torch.tensor(X_np[train_index], dtype=torch.double)
            y_train_fold = torch.tensor(y_np[train_index], dtype=torch.double)
            X_val_fold = torch.tensor(X_np[val_index], dtype=torch.double)
            y_val_fold = torch.tensor(y_np[val_index], dtype=torch.double)
            
            #2. Re-initialize model weights for a fair comparison in each fold
            #Note: This is crucial for proper CV to avoid bias from previous folds.
            #We rely on the model object having a 'seed' and 'X' attribute access.
            new_model = LinearRegression(self.model_obj.name, X_train_fold, 234 + fold)
            new_trainer = Trainer(self.name, new_model, self.eta, self.n_iter)
            
            #3. Train the model instance for the current fold
            new_trainer.training(X_train_fold, y_train_fold)
        
            #4. Evaluate the model on the validation fold
            with torch.no_grad():
                output = new_model.predict(X_val_fold)
                val_loss = criterion(output, y_val_fold).item()
            
            all_fold_losses.append(val_loss)
            print(f'Validation Loss: {val_loss:.6f}')

        avg_loss = np.mean(all_fold_losses)
        print(f'\nAverage CV Loss across {n_splits} folds: {avg_loss:.6f}')
        return avg_loss, all_fold_losses

    #--------------------------------------------------------------------------------------------------------
    #End of Cross-Validation Method
    #--------------------------------------------------------------------------------------------------------
    

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
        custom_dataset = TensorDataset(X,y)
        train_loader = DataLoader(custom_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.avg_losses = [] # losses array initialisation
        criterion = nn.MSELoss() # loss function Mean Square Error

        self.optim_selection()
        # training routine
        for i in range(self.n_iter):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                self.optimiser.zero_grad() # zeroing gradients
                output = self.model_obj.net_input(inputs) 
                loss = criterion(output,labels)
                loss.backward() # calculate gradients
                self.optimiser.step() # updating parameters
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            self.avg_losses.append(avg_loss)
            print(f"Iter [{i+1}/{self.n_iter}], Loss: {avg_loss:.4f}")
        print("Training complete.")
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


