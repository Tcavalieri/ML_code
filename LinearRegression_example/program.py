from FeaturesExtraction import FeaturesExtraction
from Models import LinearRegression
from Trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
from InputReader import InputReader
from Logger import Logger


# Reading the input file (*.yml file)
inp = InputReader('input.yml')
inp_read = inp.yml_reader()

# Extracting features and labels
features_extract = FeaturesExtraction(inp_read.content['datafile_name']) # creating class instance
features_extract.data_analysis() # storing information of the database
[labs, feats] = features_extract.feature_extraction(inp_read.content['label'],inp_read.content['features']) # extraction of features and labels

# Model
lr_model1 = LinearRegression(inp_read.content['model'],feats,inp_read.content['random_state']) # instance of a linear regression model
lr_model2 = LinearRegression(inp_read.content['model'],feats,inp_read.content['random_state']) # other instance only for demonstartion purposes

# Training
trainer1 = Trainer(inp_read.content['trainer'],lr_model1,inp_read.content['eta'],inp_read.content['n_iter']) # instance of training class
trainer2 = Trainer(inp_read.content['trainer'],lr_model2,inp_read.content['eta'],inp_read.content['n_iter']) # other instance only for demonstartion purposes

# First part of the log file
log = Logger(lr_model1,trainer1)
log.log('w')

train1 = trainer1.training(feats,labs) # stochastic gradient descent training
train2 = trainer2.gd_optim(feats, labs) # gradient descent training

# Second part of the log file
log.log('a')

# Plot of the training

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# example of a stochastic gradiesnt descent optimisation algorithm
ax[0].plot(range(1, len(train1.losses) + 1),np.log10(train1.losses), marker='o')
ax[0].set_xlabel('Number of iter')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title(f'Stocastic gradient descent')
# example of a linear gradiesnt descent optimisation algorithm
ax[1].plot(range(1, len(trainer2.losses) + 1),np.log10(trainer2.losses), marker='o')
ax[1].set_xlabel('Number of iter')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title(f'gradient descent')
plt.show()

# Prediction of the model trained with stochastic gradient descent
prediction = lr_model1.predict(feats)

plt.scatter(labs,prediction.detach().numpy())
plt.plot(np.linspace(0,70),np.linspace(0,70), color='r')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.grid()
plt.show()



