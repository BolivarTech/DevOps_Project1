'''
This library provide Customer Churn Predictions based on model

Author: Julian Bolvar
Version: 1.0.0
Date history:
2023-04-22  Refactory Done and Standard Coding Passed
2023-04-16  Base code
'''

# PArguments Parser
from argparse import ArgumentParser
# import system libraries
import os
import sys
import logging.handlers
import logging as log
# ML libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# import system libraries

# Set OS enviroment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Main LOGGER Environment
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO
LOGFILENAME = 'churn_library.log'

# OS running
OS_ = 'unknown'


class ChurnPredictor:
    '''
    Class that implements the Churn Predictor algorimts
    '''

    def __init__(self, dataPth, imgPth='./images', modelsPth='./models',
                 log_handler=None):
        '''
        class constructor

        Inputs:
            imgPth (str): Path where the images are going to be saved
            modelsPth (str): Path where models are going to be saved
            log_handler: Loggin manager handdler
        '''

        #global LOGLEVEL_

        # Set the error LOGGER
        self._logger = log.getLogger('churn_predictor_class')
        # Add handler to LOGGER
        self.__loghandler = log_handler
        if self.__loghandler is not None:
            self._logger.addHandler(self.__loghandler)
        else:
            self._logger.debug("LOGHANDLER NOT defined (001)")
        # Set LOGGER Lever
        self._logger.setLevel(LOGLEVEL_)

        if dataPth is not None:
            try:
                self._df = pd.read_csv(dataPth)
            except FileNotFoundError as err:
                self._logger.error(err)
            else:
                self._logger.info(
                    "File %s loaded with shape %s (002)",
                    dataPth,
                    self._df.shape)
                self._logger.debug("%s", self._df.head())
        else:
            self._df = None
            self._logger.warning("NO Dataset File Defined (003)")

        if imgPth is not None:
            self._img_pth = imgPth
            self._logger.debug("Image Path set to %s (004)", self._img_pth)
        else:
            self._img_pth = "."
            self._logger.warning("Image Path NOT Defined (005)")

        if modelsPth is not None:
            self._models_pth = modelsPth
            self._logger.debug("Models Path set to %s (006)", self._models_pth)
        else:
            self._models_pth = "."
            self._logger.warning("Models Path NOT Defined (007)")

        # Model Variables
        self._run_model = None
        self._x = None
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None
        self.__rfc = None
        self.__lrc = None
        self.__cv_rfc = None
        self._y_train_preds_rf = None
        self._y_test_preds_rf = None
        self._y_train_preds_lr = None
        self._y_test_preds_lr = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth (str): a path to the csv
        output:
                df: pandas dataframe
        '''

        if pth is not None:
            try:
                self._df = pd.read_csv(pth)
            except FileNotFoundError as err:
                self._logger.error("%s (008)", err)
                raise err
            else:
                self._logger.info(
                    "File %s loaded with shape %s (009)",
                    pth,
                    self._df.shape)
                self._logger.debug("%s (010)", self._df.head())
        else:
            self._df = None
            self._logger.warning("NO Dataset File Defined (011)")
        return self._df

    def perform_eda(self):
        '''
        perform EDA on df and save figures to images folder
        input:
                None

        output:
                None
        '''

        self._logger.info("\n%s (012)", self._df.isnull().sum())
        self._logger.debug("\n%s  (013)", self._df.describe())

        self._df['Churn'] = self._df['Attrition_Flag']. \
            apply(lambda val: 0 if val == "Existing Customer"
                  else 1)

        plt.figure(figsize=(20, 10))
        # plot = self._df['Churn'].hist()
        self._df['Churn'].hist()
        # Save the histogram
        plt.savefig(self._img_pth + "/" + 'hist-1.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        #plot = self._df['Customer_Age'].hist()
        self._df['Customer_Age'].hist()
        plt.savefig(self._img_pth + "/" + 'hist-2.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        # plot = self._df.Marital_Status.value_counts('normalize') \
        #    .plot(kind='bar')
        self._df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(self._img_pth + "/" + 'hist-3.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(self._df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(self._img_pth + "/" + 'hist-4.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(self._df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(self._img_pth + "/" + 'hist-5.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        sns.heatmap(self._df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        # plt.show()
        plt.savefig(self._img_pth + "/" + 'hist-6.png')
        plt.close()

    def encoder_helper(self, category_lst, response='Churn'):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 16 from the
        notebook

        input:
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could
                        be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for each category
        '''
        for category in category_lst:
            groups = self._df.groupby(category).mean()[response]
            self._df[category + "_" + response] = [groups.loc[val] for val in
                                                   self._df[category]]

        return self._df

    def perform_feature_engineering(self, response='Churn'):
        '''
        input:
                response: string of response name [optional argument that could
                        be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        y_values = self._df[response]

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

        self._x = pd.DataFrame()
        self._x[keep_cols] = self._df[keep_cols]
        self._logger.debug("%s (014)", self._x.head())

        # train test split
        self._x_train, \
            self._x_test, \
            self._y_train, \
            self._y_test = train_test_split(self._x, y_values, test_size=0.3,
                                            random_state=42)

        return (self._x_train, self._x_test, self._y_train, self._y_test)

    def train_models(self):
        '''
        train, store model results: images + scores, and store best models in
        ./models directory
        input:
                None
        output:
                None
        '''

        self._logger.info("Training Begining (015)")
        # grid search
        self.__rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html
        # logistic-regression
        self.__lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        self.__cv_rfc = GridSearchCV(estimator=self.__rfc,
                                     param_grid=param_grid, cv=5)
        self.__cv_rfc.fit(self._x_train, self._y_train)

        self.__lrc.fit(self._x_train, self._y_train)

        self._y_train_preds_rf = self.__cv_rfc.best_estimator_ \
            .predict(self._x_train)
        self._y_test_preds_rf = self.__cv_rfc.best_estimator_ \
            .predict(self._x_test)

        self._y_train_preds_lr = self.__lrc.predict(self._x_train)
        self._y_test_preds_lr = self.__lrc.predict(self._x_test)

        # scores
        logout = "random forest results\n"
        logout += "test results\n"
        logout += classification_report(self._y_test, self._y_test_preds_rf) \
            + "\n"
        logout += "train results\n"
        logout += classification_report(self._y_train,
                                        self._y_train_preds_rf) + "\n"

        logout += "logistic regression results\n"
        logout += "test results\n"
        logout += classification_report(self._y_test, self._y_test_preds_lr) \
            + "\n"
        logout += "train results\n"
        logout += classification_report(self._y_train,
                                        self._y_train_preds_lr) + "\n"

        self._logger.info("%s (016)", logout)

        # save best model
        joblib.dump(self.__cv_rfc.best_estimator_, self._models_pth
                    + '/rfc_model.pkl')
        joblib.dump(self.__lrc, self._models_pth + '/logistic_model.pkl')
        self._logger.info("Training Finished (017)")

    def __reports_plots_saver(self, model, model_name):
        '''
        Save model plots reports

        Inputs:
            model: model to save
            model_name: Model Name

        Outputs:
            None
        '''

        model_plot = plot_roc_curve(model, self._x_test, self._y_test)
        plt.savefig(
            f"{self._img_pth}/false-true-positives_rate_{model_name}-best-model.png")
        plt.close()

        plt.figure(figsize=(15, 8))
        ax_values = plt.gca()
        model_plot.plot(ax=ax_values, alpha=0.8)
        plt.savefig(
            f"{self._img_pth}/false-true-positives_rate_{model_name}-best-model.png")
        plt.close()

    def classification_report_image(self):
        '''
        produces classification report for training and testing results and
        stores report as image in images folder.

        input:
                None

        output:
                None
        '''

        self._logger.info("Classification Report Started (018)")
        lrc_plot = plot_roc_curve(self.__lrc, self._x_test, self._y_test)
        plt.savefig(f"{self._img_pth}/false-true-positives_rate_lr.png")
        plt.close()

        plt.figure(figsize=(15, 8))
        ax_values = plt.gca()
        lrc_plot.plot(ax=ax_values, alpha=0.8)
        plt.savefig(f"{self._img_pth}/false-true-positives_rate_rfc.png")
        plt.close()

        # Test Saved Models
        rfc_model = joblib.load(self._models_pth + '/rfc_model.pkl')
        self.__reports_plots_saver(rfc_model, "rfc")

        lr_model = joblib.load(f"{self._models_pth}/logistic_model.pkl")
        self.__reports_plots_saver(lr_model, "lr")

        self._logger.debug("to run Mean Shap (019)")
        explainer = shap.TreeExplainer(self.__cv_rfc.best_estimator_)
        self._logger.debug("to get Shap values (020)")
        shap_values = explainer.shap_values(self._x_test)
        self._logger.debug("to Plot Mean Shap (021)")
        shap.summary_plot(shap_values, self._x_test, plot_type="bar",
                          show=False)
        self._logger.debug("to save plot Mean Shap (022)")
        plt.savefig(self._img_pth + "/" + 'mean_SHAP.png')
        plt.close()

        self._logger.debug("Mean Shap done (023)")
        self._logger.info("Classification Report Finished (024)")

    def feature_importance_plot(self):
        '''
        creates and stores the feature importances in pth

        input:
                None
        output:
                None
        '''

        self._logger.info("Features Importance Report Started (025)")
        # Calculate feature importances
        importances = self.__cv_rfc.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self._x.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self._x.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self._x.shape[1]), names, rotation=90)
        plt.savefig(self._img_pth + "/" + 'features_importance.png')
        plt.close()

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, 'Random Forest Train', {'fontsize': 10},
                 fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self._y_test, self._y_test_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, 'Random Forest Test', {'fontsize': 10},
                 fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self._y_train, self._y_train_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(self._img_pth + "/" + 'random_forest.png')
        plt.close()

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, 'Logistic Regression Train', {
            'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self._y_train, self._y_train_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, 'Logistic Regression Test', {'fontsize': 10},
                 fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self._y_test, self._y_test_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(self._img_pth + "/" + 'logistic_regression.png')
        plt.close()

        self._logger.info("Features Importance Report Finished (026)")

    def load_model(self, model_path_file):
        '''
        Load the model in model_path_file to be runned on production

        Inputs:
            model_path_file: File with the model to be loaded

        Outputs:
            None
        '''

        self._logger.info("Loading Model %s", model_path_file)
        try:
            self._run_model = joblib.load(model_path_file)
        except FileNotFoundError as err:
            self._logger.error("%s (027)", err)
            raise err
        else:
            self._logger.info("Model %s Loaded", model_path_file)

    def predict(self, x_data=None):
        '''
        Predict the Y values based on the loaded model

        Input:
            x_data: {array-like, sparse matrix} of shape (n_samples, n_features)
                     The data matrix for which we want to get the predictions.
        Outputs:
            y_data: array of shape (n_samples,) containing the class labels for
                    each sample.
        '''

        if x_data is None:
            x_data = self._x
        return self._run_model.predict(x_data)

    def run(self, mode='Train', model_path_file=None):
        '''
        Run the Classes Implementations

        Input:
            mode: especify if is excecuting on 'Train' mode or 'Run' mode.
            model_path_file: Path to the model File
        Outputs:
            None
        '''

        LOGGER.info("Script Execution Began (028)")
        if mode == 'Train':
            self.perform_eda()
            self.perform_feature_engineering()
            self.train_models()
            self.classification_report_image()
            self.feature_importance_plot()
        elif mode == 'Run':
            self.load_model(model_path_file)
            self.perform_eda()
            self.perform_feature_engineering()
            y_data = self.predict()
            print(y_data)
        else:
            LOGGER.error("Script Unkown Excecution mode (029)")
        LOGGER.info("Script Execution Finished (030)")


def build_argparser():
    """
    Parse command line arguments.

    Inputs:
        None

    Outputs:
        command line arguments
    """

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        required=False,
        action="store_true",
        help="Perform a model training, if -a "
        "not specified a new model is "
        "trained.")
    parser.add_argument(
        "-r",
        "--run",
        required=False,
        action="store_true",
        help="Perform the model running, -m must be especified .")
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default='./models/logistic_model.pkl',
        help="Model to be loaded on running mode.")
    parser.add_argument(
        "-d",
        "--data",
        required=False,
        type=str,
        default='./data/bank_data.csv',
        help="Data to be used on running or trainign mode.")
    return parser


def main():
    '''
    Main script function

    Inputs:
        None

    Outputs:
        None
    '''

    args = build_argparser().parse_args()
    data = args.data
    model = args.model
    if args.train and args.run:
        LOGGER.error("Options Train and Run can't be used togethers (031)")
    elif args.train:
        if data is not None:
            LOGGER.debug("Loading Data file %s", data)
            churn_pred = ChurnPredictor(data, log_handler=LOGHANDLER)
            churn_pred.run()
        else:
            LOGGER.error("NO data file especified (032)")
    elif args.run:
        if data is not None:
            churn_pred = ChurnPredictor(data, log_handler=LOGHANDLER)
            if model is not None:
                churn_pred.run(mode='Run', model_path_file=model)
            else:
                LOGGER.error("Model file NOT especified (033)")
        else:
            LOGGER.error("NO data file especified (034)")
    else:
        LOGGER.error("Unkwoun Running Mode (035)")


if __name__ == '__main__':
    # Excecute the Library as stand alone script

    computer_name = os.environ['COMPUTERNAME']
    LOGGPATH = "./logs"
    FULLFILENAMEPATH = f"{LOGGPATH}/{LOGFILENAME}"
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
        # LOGGPATH = '/var/log/DNav'
        LOGGPATH = './log'
        FULLFILENAMEPATH = LOGGPATH + '/' + LOGFILENAME
    elif OS_ == 'win32':
        # LOGGPATH = os.getenv('LOCALAPPDATA') + '\\DNav'
        LOGGPATH = '.\\log'
        FULLFILENAMEPATH = LOGGPATH + '\\' + LOGFILENAME

    # Configure the LOGGER
    os.makedirs(LOGGPATH, exist_ok=True)  # Create log path

    LOGGER = log.getLogger('ChurnPred')  # Get Logger
    # Add the log message file handler to the logger
    LOGHANDLER = log.handlers.RotatingFileHandler(
        FULLFILENAMEPATH, maxBytes=10485760, backupCount=5)
    # LOGGER Formater
    logFormatter = log.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    LOGHANDLER.setFormatter(logFormatter)
    # Add handler to LOGGER
    if 'LOGHANDLER' in globals():
        LOGGER.addHandler(LOGHANDLER)
    else:
        LOGGER.debug("LOGHANDLER NOT defined (036)")
    # Set LOGGER Lever
    LOGGER.setLevel(LOGLEVEL_)
    # Start Running
    LOGGER.debug("Running in %s (037)", OS_)

    # Main Script
    main()
