from FeaturesExtraction import FeaturesExtraction
from LinearRegressionGD import LinearRegressionGD
import numpy as np
import matplotlib.pyplot as plt

features_extract = FeaturesExtraction('LinearRegression_example/data.csv')
##File path had to be adjusted to run in local environment - may need to be changed back for other users


features_extract.data_analysis()



names = ['BET-sa-(m2/g)','pore-volume-(cm3/g)','T-(K)','P-(bar)']
#names = ['pore-volume-(cm3/g)'] #['BET-sa-(m2/g)'] #,'pore-volume-(cm3/g)','T-(K)','P-(bar)']
label_name = 'CO2-uptake-(mmol/g)'

#[labs, feats] = features_extract.feature_extraction(label_name,names)
#[labs2, feats2] = features_extract.feature_extraction(label_name,names)

X_train, y_train, X_test, y_test = features_extract.feature_extraction(label_name,names,test_split=0.8)
##new call to function with test_split argument, set with 0.8 for 80/20 split

#print(feats)
#print(labs)
eta1 = 1e-8
eta2 = 1e-1
n_iter = 30

#print(np.array([0.]))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

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
ax[1].set_xlabel('Number of iter')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title(f'Learning rate {eta2}')
plt.show()


# prediction of the model trained with a "good" learning rate
#prediction = test1.predict(feats)
prediction = test1.predict(X_test) ##changed to predict on unseen testing data

#plt.scatter(labs,prediction)
plt.scatter(y_test,prediction) ##compare true vs predicted for test data
plt.plot(np.linspace(0,70),np.linspace(0,70), color='r')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.grid()
plt.show()

##final syntax "RuntimeWarning: overflow ecountered in square" - likely due to large learning rate from eta2
##
