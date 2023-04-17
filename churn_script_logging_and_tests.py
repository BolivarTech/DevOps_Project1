"""
Perform TDD on the classes
"""
from os.path import exists
import os
import logging
from churn_library import churn_predictor 


class churn_predictor_test(churn_predictor):
    '''
    Class that perform the churn predictor's test
    '''

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
    
        try:
            assert self.__df.shape[0] > 0, "Testing import_data: The file \
            doesn't appear to have columns"  
            assert self.__df.shape[1] > 0, "Testing import_data: The file \
            doesn't appear to have rows"
        except AssertionError as err:
            logging.error(err)
            raise err
    
    
    def test_eda(self):
        '''
        test perform eda function
        '''
        self.perform_eda()
        
		# Test that all files were created
        files = [self.__imgPth + "/" + 'hist-' + i + '.png' for i in range(1,7)]
        try:
            for file in files:
                assert exists(file), f"File {file} doen't exist"
        except AssertionError as err:
            logging.error(err)
    
    
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
        org_colums = list(self.__df.columns)
        
        self.encoder_helper(cat_columns, response)
        
		# Modified Colums
        new_colums = list(self.__df.columns)
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
    

    
    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''

        self.perform_feature_engineering()
        
		# Tests
        try: 
            assert hasattr(self,'__X_train'), f"__X_train not defined"
            assert hasattr(self,'__X_test'), f"__X_test not defined"
            assert hasattr(self,'__y_train'), f"__y_train not defined"
            assert hasattr(self,'__y_test'), f"__y_test not defined"
        except AssertionError as err:
            logging.error(err)
    
    
    def test_train_models(self):
        '''
        test train_models
        '''
        
        self.train_models()
        
		# Test that all files were created
        files = [self.__modelsPth + '/rfc_model.pkl', self.__modelsPth + '/logistic_model.pkl']
        try:
            for file in files:
                assert exists(file), f"File {file} doen't exist"
        except AssertionError as err:
            logging.error(err)



if __name__ == "__main__":
	logging.basicConfig(
        filename='./logs/churn_library_test.log',
        level = logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')
    
	# Run Tests
	tester = churn_predictor_test("./data/bank_data.csv")
