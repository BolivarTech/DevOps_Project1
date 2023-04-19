"""
Perform TDD on the classes
"""
# import system libraries
import logging as log
import logging.handlers
import sys
import os
from os.path import exists

from churn_library import churn_predictor 

# Main Logger Environment 
logHandler = None
logger = None
logLevel_ = logging.DEBUG
logFileName = 'churn_library_test.log'

class churn_predictor_test(churn_predictor):
    '''
    Class that perform the churn predictor's test
    '''

    def _test_files_exist(self, files, testName='None'):
        '''
        Test if files list exist
        '''

        try:
            for file in files:
                assert exists(file), f"File {file} doen't exist"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = f"Test {testName} Passed"
            print(msg)
            logging.info(msg)


    def test_import(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        try:
            self.import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found\n {err}")
            raise err
        else:
            msg = f"Test Import Passed"
            print(msg)
            logging.info(msg)
    
        try:
            assert hasattr(self,'_df'), f"_df not defined"
            assert self._df.shape[0] > 0, "Testing import_data: The file \
            doesn't appear to have columns"  
            assert self._df.shape[1] > 0, "Testing import_data: The file \
            doesn't appear to have rows"
        except AssertionError as err:
            logging.error(err)
            raise err
        else:
            msg = f"Test _df Passed"
            print(msg)
            logging.info(msg)
    
    def test_eda(self):
        '''
        test perform eda function
        '''
        self.perform_eda()
        
		# Test that all files were created
        files = [self._imgPth + "/" + 'hist-' + str(i) + '.png' \
                for i in range(1,7)]
        self._test_files_exist(files,testName='eda')
    
    
    def test_encoder_helper(self):
        '''
        test encoder helper
        '''
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'                
        ]
        
        response='Churn'

		# Original Colums
        org_colums = list(self._df.columns)
        
        self.encoder_helper(cat_columns, response)
        
		# Modified Colums
        new_colums = list(self._df.columns)
        new_cat_columns = [category + "_" + response for category in cat_columns]
        
		# Tests
        try:
            assert (len(new_colums) == (len(org_colums) + len(cat_columns))), \
                    f"Colums size dosen't match {len(new_colums)} vs \
                    {(len(org_colums) + len(cat_columns))}"
            assert set(new_cat_columns).issubset(set(new_colums)), \
                    f"Columns names doesn't match\n {new_cat_columns}\n \
                    {new_colums}"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = f"Test encoder_helper Passed"
            print(msg)
            logging.info(msg)

    
    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''

        self.perform_feature_engineering()
        
		# Tests
        try: 
            assert hasattr(self,'_X_train'), f"_X_train not defined"
            assert hasattr(self,'_X_test'), f"_X_test not defined"
            assert hasattr(self,'_Y_train'), f"_Y_train not defined"
            assert hasattr(self,'_Y_test'), f"_Y_test not defined"
        except AssertionError as err:
            logging.error(err)
        else:
            msg = f"Test feature_engineering Passed"
            print(msg)
            logging.info(msg)
    
    def test_train_models(self):
        '''
        test train_models
        '''
        
        self.train_models()
        
		# Test that all files were created
        files = [self._modelsPth + '/rfc_model.pkl', 
                self._modelsPth + '/logistic_model.pkl']
        self._test_files_exist(files,testName='train_model')
        

    def test_classification_report_image(self):
        '''
        Test if classification images reports where created
        '''

        self.classification_report_image()

        # Test that all files were created
        files = [self._imgPth + "/" + 'false-true-positives_rate_lrc.png',
                self._imgPth + "/" + 'false-true-positives_rate_rfc.png',
                self._imgPth + "/" 
                + 'false-true-positives_rate_lrc-best-model.png',
                self._imgPth + "/" 
                + 'false-true-positives_rate_rfc-best-model.png',
                self._imgPth + "/" + 'mean_SHAP.png'
                ]
        self._test_files_exist(files, testName='classification_report_images')


    def test_feature_importance_plot(self):
        '''
        Test if feature importaace plors where created
        '''

        self.feature_importance_plot()

        # Test that all files were created
        files = [self._imgPth + "/" + 'features_importance.png',
                self._imgPth + "/" + 'random_forest.png',
                self._imgPth + "/" + 'logistic_regression.png'
                ]
        self._test_files_exist(files, testName='feature_importance_plot')
        

    def run(self):
        '''
        Run the Test
        '''

        self.test_import()
        self.test_eda()
        self.test_encoder_helper()
        self.test_perform_feature_engineering()
        self.test_train_models()
        self.test_classification_report_image()
        self.test_feature_importance_plot()

if __name__ == "__main__":
	# logging.basicConfig(
    #     filename='./logs/churn_library_test.log',
    #     level = logging.INFO,
    #     filemode='w',
    #     format='%(name)s - %(levelname)s - %(message)s')
    
    logger = log.getLogger('ChurmPredTest')  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler('./log/' + logFileName, maxBytes=10485760, backupCount=5)
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
    
	# Run Tests
    tester = churn_predictor_test("./data/bank_data.csv", log_handler=logHandler)
    tester.run()
