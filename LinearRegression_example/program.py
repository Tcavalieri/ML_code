from FeaturesExtraction import FeaturesExtraction
from Models import LinearRegression
from Trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import sklearn


from InputReader import InputReader
from Logger import Logger


# Reading the input file (*.yml file)
inp = InputReader('input.yml')
inp_read = inp.yml_reader()
eta = inp_read.content['eta']
n_iter = inp_read.content['n_iter']
random_state = inp_read.content['random_state']

# Extracting features and labels
features_extract = FeaturesExtraction(inp_read.content['datafile_name']) # creating class instance
features_extract.data_analysis() # storing information of the database
[labs_np, feats_np] = features_extract.feature_extraction(inp_read.content['label'],inp_read.content['features']) # extraction of features and labels

# 1. Convert full dataset NumPy arrays to Tensors for Cross-Validation (CV)
# Uses the full dataset (test_ratio=0.0) converted to Tensors.
X_full, _, Y_full, _ = features_extract.data_split(
    feats_np, labs_np, 
    test_ratio=0.0, 
    random_state=random_state
)

#Train Test Split Implementation; Split and Tensor Conversion
X_train, X_test, y_train, y_test = features_extract.data_split(
    feats_np, labs_np, 
    test_ratio=inp_read.content['test_split_ratio'], 
    random_state=inp_read.content['random_state']
)

# Model
lr_model1 = LinearRegression(inp_read.content['model'], X_train, random_state) # instance of a linear regression model
lr_model2 = LinearRegression(inp_read.content['model'],X_train, random_state) # other instance only for demonstartion purposes

# Training
trainer1 = Trainer(inp_read.content['trainer'],lr_model1,inp_read.content['eta'],inp_read.content['n_iter']) # instance of training class
trainer2 = Trainer('GD_Custom',lr_model2,inp_read.content['eta'],inp_read.content['n_iter'])

# --- START CROSS-VALIDATION ---
print("\n*** Starting Cross-Validation for SGD Model Performance Estimate ***")
# Note: trainer1 model is repeatedly re-initialized inside cross_validate for fairness
avg_cv_loss, cv_losses = trainer1.cross_validate(X_full, Y_full, n_splits=5)
print(f"Final Average Cross-Validation Loss (5-Fold MSE): {avg_cv_loss:.6f}")
# --- END CROSS-VALIDATION ---

#[labs, feats] = features_extract.feature_extraction(label_name,names)
#[labs2, feats2] = features_extract.feature_extraction(label_name,names)

log = Logger(lr_model1,trainer1)
log.log('w')

train1 = trainer1.training(X_train, y_train) # stochastic gradient descent training
train2 = trainer2.gd_optim(X_train, y_train) # gradient descent training

# Second part of the log file
log = Logger(lr_model1, trainer1)
log.log('a')

# Plot of the training
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

#Plot SDG Loss, using results from the train1
ax[0].plot(range(1, len(train1.losses) + 1), np.log10(train1.losses), marker='o')
ax[0].set_xlabel('Number of iter')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title(f'Stocastic gradient descent')

#Plot GD Loss, using results from train2
ax[1].plot(range(1, len(train2.losses) + 1), np.log10(train2.losses), marker='o')
ax[1].set_xlabel('Number of iter')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title(f'gradient descent')
plt.show()

# prediction of the model trained with a "good" learning rate
#prediction = test1.predict(feats)
prediction_test = lr_model1.predict(X_test) ##changed to predict on unseen testing data

#Convert Tensors to Numpy for plotting
y_test_np = y_test.detach().numpy()
prediction_test_np = prediction_test.detach().numpy()

#plt.scatter(labs,prediction)
plt.scatter(y_test_np,prediction_test_np) ##compare true vs predicted for test data

#Compare true vs predicted values for test data
plt.scatter(y_test_np, prediction_test_np)

#Plot the 1:1 line for visual reference: this line represents where the prediction = the true value
plt.plot(np.linspace(min(y_test_np), max(y_test_np)), np.linspace(min(y_test_np), max(y_test_np)), color='r')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.grid()
plt.title('Prediction vs True Values on Test Data, SGD Trained Model')
plt.show()

'''
# Prediction of the model trained with stochastic gradient descent
prediction = lr_model1.predict(feats)

plt.scatter(labs,prediction.detach().numpy())
plt.plot(np.linspace(0,70),np.linspace(0,70), color='r')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.grid()
plt.show()
'''

'''
# example of a training with a "good" learning rate
#test1 = LinearRegressionGD(eta1,n_iter).fit(feats,labs) --- original line
test1 = LinearRegressionGD(eta1, n_iter).fit(X_train, y_train) ##changed to use training data from split
ax[0].plot(range(1, len(test1.losses_) + 1),np.log10(test1.losses_), marker='o')
ax[0].set_xlabel('Number of iter')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title(f'Learning rate {eta1}')

# example of a training with a "bad" learning rate
#test2 = LinearRegressionGD(eta2, n_iter).fit(feats2, labs2) --- original line
test2 = LinearRegressionGD(eta2, n_iter).fit(X_train, y_train) ##changed to use training data from split
ax[1].plot(range(1, len(test2.losses_) + 1),np.log10(test2.losses_), marker='o')
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
'''


