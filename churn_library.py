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
    
    def __init__(self, dataPth, imgPth='./images', modelsPth='./models', log_handler=None):
        '''
        class constructor
        '''
        
        global logLevel_

        # Set the error logger
        self.__logger = log.getLogger('churn_predictor_class')
        # Add handler to logger
        self.__logHandler = log_handler
        if self.__logHandler is not None:
            self.__logger.addHandler(self.__logHandler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)


        if dataPth is not None:
            try:
                self._df = pd.read_csv(dataPth)
            except FileNotFoundError as err:
                self.__logger.error(err)
            else:
                self.__logger.info(f"File {dataPth} loaded with shape {self._df.shape} (01)")
                self.__logger.debug(f"{self._df.head()}")
        else:
            self._df = None
            self.__logger.warning(f"NO Dataset File Defined (02)")
        
        if imgPth is not None:
            self._imgPth = imgPth
            self.__logger.debug(f"Image Path set to {self._imgPth} ()")
        else:
            self._imgPth = "."
            self.__logger.warning(f"Image Path NOT Defined (02)")

        if modelsPth is not None:
            self.__modelsPth = modelsPth
            self.__logger.debug(f"Models Path set to {self.__modelsPth} ()")
        else:
            self.__modelsPth = "."
            self.__logger.warning(f"Models Path NOT Defined (02)")

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth
    
        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''	

        if pth is not None:
            try:
                self._df = pd.read_csv(pth)
            except FileNotFoundError as err:
                self.__logger.error(err)
                raise err
            else:
                self.__logger.info(f"File {pth} loaded with shape {self._df.shape} (03)")
                self.__logger.debug(f"{self._df.head()}")
        else:
            self._df = None
            self.__logger.warning(f"NO Dataset File Defined (04)")
        return self._df

    
    
    def perform_eda(self):
        '''
        perform EDA on df and save figures to images folder
        input:
                None
    
        output:
                None
        '''
        
        self.__logger.info(f"{self._df.isnull().sum()}")
        self.__logger.debug(f"{self._df.describe()}")

        quant_columns = [
            'Customer_Age',
            'Dependent_count', 
            'Months_on_book',
            'Total_Relationship_Count', 
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 
            'Credit_Limit', 
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 
            'Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt',
            'Total_Trans_Ct', 
            'Total_Ct_Chng_Q4_Q1', 
            'Avg_Utilization_Ratio'
        ]
    
        self._df['Churn'] = self._df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        plt.figure(figsize=(20,10)) 
        plot = self._df['Churn'].hist()
        # Save the histogram
        plt.savefig(self._imgPth + "/" + 'hist-1.png')

        plt.figure(figsize=(20,10)) 
        plot = self._df['Customer_Age'].hist()
        plt.savefig(self._imgPth + "/" + 'hist-2.png')

        plt.figure(figsize=(20,10)) 
        plot = self._df.Marital_Status.value_counts('normalize').plot(kind='bar');
        plt.savefig(self._imgPth + "/" + 'hist-3.png')

        plt.figure(figsize=(20,10)) 
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
        sns.histplot(self._df['Total_Trans_Ct'], stat='density', kde=True);
        plt.savefig(self._imgPth + "/" + 'hist-4.png')

        plt.figure(figsize=(20,10)) 
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
        sns.histplot(self._df['Total_Trans_Ct'], stat='density', kde=True);
        plt.savefig(self._imgPth + "/" + 'hist-5.png')

        plt.figure(figsize=(20,10)) 
        sns.heatmap(self._df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        # plt.show()
        plt.savefig(self._imgPth + "/" + 'hist-6.png')


    def encoder_helper(self, category_lst, response='Churn'):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 16 from the 
        notebook
    
        input:
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]
    
        output:
                df: pandas dataframe with new columns for each category
        '''
        for category in category_lst:
            groups = self._df.groupby(category).mean()[response]
            self._df[category + "_" + response] = [ groups.loc[val] for val in self._df[category] ]     
            return self._df
    
    
    def perform_feature_engineering(self, response='Churn'):
        '''
        input:
                response: string of response name [optional argument that could be used for naming variables or index y column]
    
        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        y = self._df[response]
        
        category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category']
        
        self.encoder_helper(category_lst)

        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 
                    'Total_Revolving_Bal', 'Avg_Open_To_Buy', 
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 
                    'Avg_Utilization_Ratio', 'Gender_Churn', 
                    'Education_Level_Churn', 'Marital_Status_Churn', 
                    'Income_Category_Churn', 'Card_Category_Churn']

        self.__X = pd.DataFrame()
        self.__X[keep_cols] = self._df[keep_cols]
        self.__logger.debug(f"{self.__X.head()}")
        
        # train test split 
        self.__X_train,
        self.__X_test,
        self.__y_train,
        self.__y_test = train_test_split(self.__X, y, test_size= 0.3, random_state=42)

        return (self.__X_train, self.__X_test, self.__y_train, self.__y_test)
    

    def train_models(self):
        '''
        train, store model results: images + scores, and store best models
        input:
                None
        output:
                None
        '''
        
        # grid search
        self.__rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        self.__lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }
        
        self.__cv_rfc = GridSearchCV(estimator=self.__rfc, param_grid=param_grid, cv=5)
        self.__cv_rfc.fit(self.__X_train, self.__y_train)
        
        self.__lrc.fit(self.__X_train, self.__y_train)
        
        self.__y_train_preds_rf = self.__cv_rfc.best_estimator_.predict(self.__X_train)
        self.__y_test_preds_rf = self.__cv_rfc.best_estimator_.predict(self.__X_test)
        
        self.__y_train_preds_lr = self.__lrc.predict(self.__X_train)
        self.__y_test_preds_lr = self.__lrc.predict(self.__X_test)
        
        # scores
        logout = "random forest results\n"
        logout += "test results\n"
        logout += classification_report(self.__y_test, self.__y_test_preds_rf) + "\n"
        logout += "train results\n"
        logout += classification_report(self.__y_train, self.__y_train_preds_rf) + "\n"
        
        logout += "logistic regression results\n"
        logout += "test results\n"
        logout += classification_report(self.__y_test, self.__y_test_preds_lr) + "\n"
        logout += "train results\n"
        logout += classification_report(self.__y_train, self.__y_train_preds_lr) + "\n"

        self.__logger.info(logout)

        # save best model
        joblib.dump(self.__cv_rfc.best_estimator_, self.__modelsPth + '/rfc_model.pkl')
        joblib.dump(self.__lrc, self.__modelsPth + '/logistic_model.pkl')
    

    def classification_report_image(self):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                None
    
        output:
                None
        '''
        
        lrc_plot = plot_roc_curve(self.__lrc, self.__X_test, self.__y_test)
        plt.savefig(self._imgPth + "/" + 'false-true-positives_rate_lrc.png')
        
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(self.__cv_rfc.best_estimator_, self.__X_test, self.__y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        #plt.show()
        plt.savefig(self._imgPth + "/" + 'false-true-positives_rate_rfc.png')

        # Test Saved Models
        rfc_model = joblib.load(self.__modelsPth + '/rfc_model.pkl')
        lr_model = joblib.load(self.__modelsPth + '/logistic_model.pkl')

        lrc_plot = plot_roc_curve(lr_model, self.__X_test, self.__y_test)
        plt.savefig(self._imgPth + "/" + 'false-true-positives_rate_lrc-best-model.png')

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc_model, self.__X_test, self.__y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        #plt.show()
        plt.savefig(self._imgPth + "/" + 'false-true-positives_rate_rfc-best-model.png')

        explainer = shap.TreeExplainer(self.__cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(self.__X_test)
        shap.summary_plot(shap_values, self.__X_test, plot_type="bar")
        plt.savefig(self._imgPth + "/" + 'mean_SHAP.png')


    def feature_importance_plot(self):
        '''
        creates and stores the feature importances in pth
        input:
                NoneS
        output:
                None
        '''
        
        # Calculate feature importances
        importances = self.__cv_rfc.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [self.__X.columns[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=(20,5))
        
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        
        # Add bars
        plt.bar(range(self.__X.shape[1]), importances[indices])
        
        # Add feature names as x-axis labels
        plt.xticks(range(self.__X.shape[1]), names, rotation=90);

        plt.savefig(self._imgPth + "/" + 'features_importance.png')

        plt.rc('figure', figsize=(5, 5))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(self.__y_test, self.__y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(self.__y_train, self.__y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off');
        plt.savefig(self._imgPth + "/" + 'random_forest.png')

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(self.__y_train, self.__y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(self.__y_test, self.__y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off');
        plt.savefig(self._imgPth + "/" + 'logistic_regression.png')


    def run(self):
        '''
        '''
        self.perform_eda()
        self.perform_feature_engineering()
        self.train_models()
        self.classification_report_image()
        self.feature_importance_plot()


def main():
    '''
    Run the main script function
    '''
    churnPred = churn_predictor("./data/bank_data.csv", log_handler=logHandler)
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
    main()
