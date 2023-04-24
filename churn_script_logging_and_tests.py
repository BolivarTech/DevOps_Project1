'''
Perform TDD on the classes

Author: Julian Bolvar
Version: 1.0.0
Date history:
2023-04-22  All Tests passed completed and standard coding passed
2023-04-19  Base code
'''

# import system libraries
import logging as log
import logging.handlers
from os.path import exists

# ML imports
import numpy as np

# Main Class Import
from churn_library import ChurnPredictor

# Main Logger Environment
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO
LOGFILENAME = 'churn_library_test.log'


class ChurnPredictorTest(ChurnPredictor):
    '''
    Class that perform the churn predictor's test
    '''

    def _test_files_exist(self, files, test_name='None'):
        '''
        Test if files list exist

        Inputs:
            files (list): List of files to test
            test_name (str): String with the Test's name

        Outputs:
            None
        '''

        try:
            for file in files:
                assert exists(file), f"File {file} doen't exist"
        except AssertionError as err:
            logging.error(f"{err} (001)")
        else:
            msg = f"Test {test_name} Passed (002)"
            print(msg)
            logging.info(msg)

    def test_import(self):
        '''
        Test data import funtion

        Inputs:
            None

        Outputs:
            None
        '''
        try:
            self.import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS (003)")
        except FileNotFoundError as err:
            logging.error(
                "Testing import_eda: The file wasn't found\n {err} (004)")
            raise err
        else:
            print("Test Import Passed (005)")
            logging.info("Test Import Passed (005)")

        try:
            assert hasattr(self, '_df'), "_df not defined (006)"
            assert self._df.shape[0] > 0, "Testing import_data: The file \
            doesn't appear to have columns (007)"
            assert self._df.shape[1] > 0, "Testing import_data: The file \
            doesn't appear to have rows (008)"
        except AssertionError as err:
            logging.error(err)
            raise err
        else:
            print("Test _df Passed (009)")
            logging.info("Test _df Passed (009)")

    def test_eda(self):
        '''
        Test perform eda function

        Inputs:
            None

        Outputs:
            None
        '''

        self.perform_eda()

        # Test that all files were created
        files = [f"{self._img_pth}/hist-{str(i)}.png"
                 for i in range(1, 7)]
        self._test_files_exist(files, test_name='eda')

    def test_encoder_helper(self):
        '''
        Test encoder helper

        Inputs:
            None

        Outputs:
            None
        '''

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        response = 'Churn'

        # Original Colums
        org_colums = list(self._df.columns)

        self.encoder_helper(cat_columns, response)

        # Modified Colums
        new_colums = list(self._df.columns)
        new_cat_columns = [category + "_" +
                           response for category in cat_columns]

        # Tests
        try:
            assert (len(new_colums) == (len(org_colums) + len(cat_columns))), \
                f"Colums size dosen't match {len(new_colums)} vs \
                    {(len(org_colums) + len(cat_columns))} (010)"
            assert set(new_cat_columns).issubset(set(new_colums)), \
                f"Columns names doesn't match\n {new_cat_columns}\n \
                    {new_colums} (011)"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = "Test encoder_helper Passed (012)"
            print(msg)
            logging.info(msg)

    def test_perform_feature_engineering(self):
        '''
        Test perform_feature_engineering

        Inputs:
            None

        Outputs:
            None
        '''

        self.perform_feature_engineering()

        # Tests
        try:
            assert hasattr(self, '_x_train'), "_x_train not defined (013)"
            assert hasattr(self, '_x_test'), "_x_test not defined (014)"
            assert hasattr(self, '_y_train'), "_y_train not defined (015)"
            assert hasattr(self, '_y_test'), "_y_test not defined (016)"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = "Test feature_engineering Passed (017)"
            print(msg)
            logging.info(msg)

    def test_train_models(self):
        '''
        Test train_models

        Inputs:
            None

        Outputs:
            None
        '''

        self.train_models()

        # Test that all files were created
        files = [f"{self._models_pth}/rfc_model.pkl",
                 f"{self._models_pth}/logistic_model.pkl"]
        self._test_files_exist(files, test_name='train_model')

    def test_classification_report_image(self):
        '''
        Test if classification images reports where created

        Inputs:
            None

        Outputs:
            None
        '''

        self.classification_report_image()

        # Test that all files were created
        files = [
            f"{self._img_pth}/false-true-positives_rate_lr.png",
            f"{self._img_pth}/false-true-positives_rate_rfc.png",
            f"{self._img_pth}/false-true-positives_rate_rfc-best-model.png",
            f"{self._img_pth}/false-true-positives_rate_lr-best-model.png",
            f"{self._img_pth}/mean_SHAP.png"]
        self._test_files_exist(files, test_name='classification_report_images')

    def test_feature_importance_plot(self):
        '''
        Test if feature importaace plors where created

        Inputs:
            None

        Outputs:
            None_Y_Test_Preds_Lr
        '''

        self.feature_importance_plot()

        # Test that all files were created
        files = [f"{self._img_pth}/features_importance.png",
                 f"{self._img_pth}/random_forest.png",
                 f"{self._img_pth}/logistic_regression.png"
                 ]
        self._test_files_exist(files, test_name='feature_importance_plot')

    def test_load_model(self):
        '''
        Test the model loader

        Inputs:
            None

        Outputs:
            None
        '''

        # Tests
        try:
            self.load_model('foo')
        except Exception as err:
            if not isinstance(err, FileNotFoundError):
                logging.error("Load Module not fail on no file (018)")
                logging.error(err)
        else:
            try:
                self.load_model('./models/logistic_model.pkl')
                assert hasattr(
                    self, '_run_model'), "_run_model not defined (019)"
            except AssertionError as err:
                logging.error(err)
            except FileNotFoundError as err:
                logging.error(err)
            else:
                msg = "Test Load Model Passed (020)"
                print(msg)
                logging.info(msg)

    def test_predict(self):
        '''
        Test the model prediction function

        Inputs:
            None

        Outputs:
            None
        '''

        self.import_data("./data/bank_data.csv")
        self.load_model('./models/logistic_model.pkl')
        self.perform_eda()
        # self.test_encoder_helper()
        self.perform_feature_engineering()
        y_out = self.predict()
        try:
            assert isinstance(y_out, np.ndarray), \
                "Predict Output is not numpy.ndarray isntance (021)"
            assert y_out.size > 0, "Predict Output is not empty (022)"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = "Test Predict Passed (023)"
            print(msg)
            logging.info(msg)

    def run(self):
        '''
        Run All the Test

        Inputs:
            None

        Outputs:
            None
        '''

        print("Tests Started (024)")
        self._logger.info("Tests Started (025)")
        self.test_import()
        self.test_eda()
        self.test_encoder_helper()
        self.test_perform_feature_engineering()
        self.test_train_models()
        self.test_classification_report_image()
        self.test_feature_importance_plot()
        self.test_load_model()
        self.test_predict()
        print("Tests Finished (026)")
        self._logger.info("Tests Finished (027)")


if __name__ == "__main__":
    # Excecute the Library as stand alone script

    LOGGER = log.getLogger('ChurmPredTest')  # Get Logger
    # Add the log message file handler to the logger
    LOGHANDLER = log.handlers.RotatingFileHandler(f"./log/{LOGFILENAME}",
                                                  maxBytes=10485760,
                                                  backupCount=5)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: \
                                %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    LOGHANDLER.setFormatter(logFormatter)
    # Add handler to logger
    if 'LOGHANDLER' in globals():
        LOGGER.addHandler(LOGHANDLER)
    else:
        LOGGER.debug("LOGHANDLER NOT defined (028)")
    # Set Logger Lever
    LOGGER.setLevel(LOGLEVEL_)

    # Run Tests
    tester = ChurnPredictorTest(
        "./data/bank_data.csv",
        log_handler=LOGHANDLER)
    tester.run()
