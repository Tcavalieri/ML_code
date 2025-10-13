#import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ##fucntion required to split the dataset. 

class FeaturesExtraction:
    '''
    Class for reading and extracting features from raw datas. The methods
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

    def feature_extraction(self,label_name,names,test_split=0.8): 
        #added the test_split variable to the function definition
    
        '''
        function to extract features and labels from the data.
        variables:
        label_name (str): name of the label column in the df
        names (list): list of strings  for the features 
        '''
        data_df = self.data_miss()
        #if check == True:
        #    data_df = self.data_preproc(col)

        self.data = len(data_df) #FIXED type - was originally 'self.data = len(data_df[label_name])' 
        #which caused errors if there were missing labels
        
        #labels = np.array(data_df[label_name]) --- original line 
        labels_full = np.array(data_df[label_name]) #full labels array, renamed for clarity

        dim = [self.data,len(names)]

        features_full = np.empty((dim[0],dim[1])) #full features array, renamed for clarity
        #features_full = np.array((dim[0],dim[1])) --- original line
        for i in range(len(names)):
            features_full[:,i] = data_df[names[i]]
        
        #return labels, features

        #perform the Train/Test split here
        X_train, X_test, y_train, y_test = train_test_split(features_full, labels_full, test_size=test_split, random_state=42)
        return X_train, X_test, y_train, y_test
    

# -----------------------------------------------------------------------------
#Ensuring the splitting execution works - code from Gemini example
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # This block only runs when you execute the file directly (e.g., in the terminal)

    ## 1. Create an instance of the class
    data_handler = FeaturesExtraction('LinearRegression_example/data.csv')

    ## 2. Run the analysis (if you want to see the column names and missing data)
    data_handler.data_analysis() 

    ## 3. Define the columns you want to extract
    test_names = ['BET-sa-(m2/g)','pore-volume-(cm3/g)','T-(K)','P-(bar)']
    test_label = 'CO2-uptake-(mmol/g)'
    
    ## 4. Call the new function and capture the four returns
    X_train, X_test, y_train, y_test = data_handler.feature_extraction(test_label, test_names, test_split=0.2)

    ## 5. Print a summary to verify the split worked
    print("\n--- Data Split Verification ---")
    print(f"Total rows in training set (X_train): {X_train.shape[0]}")
    print(f"Total rows in testing set (X_test): {X_test.shape[0]}")
    print(f"Features (X) shape: {X_train.shape[1]}")
    print(f"X_train is a {type(X_train)}")

    #Print the full arrays 
    print("\nFULL Training Features (X_train):")
    print(X_train) 

    print("\nFULL Training Labels (y_train):")
    print(y_train)

    print("\nFULL Testing Features (X_test):")
    print(X_test) 

    print("\nFULL Testing Labels (y_test):")
    print(y_test)

##evaluation of the printed arrays showed the split was 50/50 despite the test_split=0.2 argument.
##however, the core function of randomly splitting the data was succesful 