import pandas as pd
import numpy as np
import yaml 

class InputReader():
    '''
    Class for reading an input file with the information for executing the ML pipeline
    '''

    def __init__(self,file_name):
        
        self.file_name = file_name
    
    def yml_reader(self):
        '''
        Method for reading *.yml input files
        '''

        with open(self.file_name, 'r') as file:
            self.content = yaml.safe_load(file)

        return self

