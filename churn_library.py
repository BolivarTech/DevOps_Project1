'''
This library provide Customer Churn Predictions based on model

Author: Julian Bolvar
Version: 1.0.0
Date history:  2023-04-16  Base code

'''
# library doc string
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


# import system libraries
import logging as log
import logging.handlers
import sys
import os

# Set OS enviroment
os.environ['QT_QPA_PLATFORM']='offscreen'

# Main Logger Environment 
logHandler = None
logger = None
logLevel_ = logging.INFO
logFileName = 'churn_library.log'

# OS running
OS_ = 'unknown'

class churn_predictor:
    
    def __init__(self, pth=None):
        '''
        class constructor
        '''
        
        if pth is not None:
            self.__df = pd.read_csv(pth)
            logger.info(f"File {pth} loaded with shape {self.__df.shape} (01)")
            logger.debug(f"{self.__df.head()}")
        else:
            self.__df = None
            logger.warning(f"NO Dataset File Defined (02)")


    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth
    
        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''	
    	
        if pth is not None:
            self.__df = pd.read_csv(pth)
            logger.info(f"File {pth} loaded with shape {self.__df.shape} (03)")
            logger.debug(f"{self.__df.head()}")
        else:
            self.__df = None
            logger.warning(f"NO Dataset File Defined (04)")
        return self.__df

    
    
    def perform_eda(self, df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe
    
        output:
                None
        '''
    	pass
    
    
    def encoder_helper(self, df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook
    
        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]
    
        output:
                df: pandas dataframe with new columns for
        '''
        pass
    
    
    def perform_feature_engineering(self, df, response):
        '''
        input:
                  df: pandas dataframe
                  response: string of response name [optional argument that could be used for naming variables or index y column]
    
        output:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        '''
    
    def classification_report_image(self,
                                    y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest
    
        output:
                 None
        '''
        pass
    
    
    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure
    
        output:
                 None
        '''
        pass
    
    def train_models(self, X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        output:
                  None
        '''
        pass
    
    def run(self):
        '''
        '''
        pass

def main():
    """
     Run the main script function

    """
    churnPred = churn_predictor()
    churnPred.run()

if __name__ == '__main__':
    '''
    Excecute the Library as stand alone script
    '''
    
    computer_name = os.environ['COMPUTERNAME']
    loggPath = "./logs"
    FullFileNamePath = loggPath + '/' + logFileName
    # Check where si running
    if sys.platform.startswith('freebsd'):
        OS_ = 'freebsd'
    elif sys.platform.startswith('linux'):
        OS_ = 'linux'
    elif sys.platform.startswith('win32'):
        OS_ = 'win32'
    elif sys.platform.startswith('cygwin'):
        OS_ = 'cygwin'
    elif sys.platform.startswith('darwin'):
        OS_ = 'darwin'
    if OS_ == 'linux':
        # loggPath = '/var/log/DNav'
        loggPath = './log'
        FullFileNamePath = loggPath + '/' + logFileName
    elif OS_ == 'win32':
        # loggPath = os.getenv('LOCALAPPDATA') + '\\DNav'
        loggPath = '.\\log'
        FullFileNamePath = loggPath + '\\' + logFileName

    # Configure the logger
    os.makedirs(loggPath, exist_ok=True)  # Create log path

    logger = log.getLogger('ChurmPred')  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler(FullFileNamePath, maxBytes=10485760, backupCount=5)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                 datefmt='%Y/%m/%d %H:%M:%S')
    logHandler.setFormatter(logFormatter)
    # Add handler to logger
    if 'logHandler' in globals():
        logger.addHandler(logHandler)
    else:
        logger.debug(f"logHandler NOT defined (017)")
    # Set Logger Lever
    logger.setLevel(logLevel_)
    # Start Running
    logger.debug(f"Running in {OS_} (018)")

    # Main Script
    main(computer_name)
