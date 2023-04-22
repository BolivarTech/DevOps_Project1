# Predict Customer Churn

- Predict Customer Churn using Machie Learning algoritms.

## Project Description
Your project description here.

## Files and data description

The project contaings the following files structure

data/ : Directory where the trainig data is store on a csv file

/data/bank_data.csv : trainig data stored on a csv file

images/ : Directory where resulting images and graft are stores

log/ : Directory where logs are stored

models/ :  Directory where trained model are strores on .pkl files

churn_library.py : Main script Library for production environmets

churn_notebook.ipynb : Jupyter Notebook development code

churn_script_logging_and_tests.py : Litrabry's Tests file

Guide.ipynb : Jupyter Notebook with project's descriptions and requirements

README.md  : This file

requirements_py3.10.txt : Dependencies for Python v3.10

requirements_py3.6.txt : Dependencies for Python v3.6

requirements_py3.8.txt : Dependencies for Python v3.8

## Requirements

This project nequired [Anaconda](https://www.anaconda.com/download)

## Instalation

After installing Anaconda, create a virtual environment:

> conda create --name churnpred python=3.8

Activate the virtual environment churnpred.

> Windows: activate churnpred  
> Linux: source activate churnpred

Install Dependencies:

> conda install --file requirements_py3.8.txt

Download Fonts requiered by QT dependency from [DeJavu](https://dejavu-fonts.github.io).

Uncompress package and install .tff files on the follow directory:

> Windows: C:\Users\[USERNAME]]\.conda\envs\churnpred\Library\lib\fonts  
> Linux: $HOME/.conda/envs/churnpred/Library/lib/fonts




## Running Files
How do you run your files? What should happen when you run your files?


### Running Script's Tests

& C:/Users/jbolivarg/.conda/envs/py38/python.exe d:/jbolivarg/JupyterLabProjects/CleanCode/project_1/churn_script_logging_and_tests.py  
Tests Started (018)  
Test Import Passed (005)  
Test _df Passed (009)  
INFO:churn_predictor_class:Unnamed: 0
CLIENTNUM                   0  
Attrition_Flag              0  
Customer_Age                0  
Gender                      0  
Dependent_count             0  
Education_Level             0  
Marital_Status              0  
Income_Category             0  
Card_Category               0  
Months_on_book              0  
Total_Relationship_Count    0  
Months_Inactive_12_mon      0  
Contacts_Count_12_mon       0  
Credit_Limit                0  
Total_Revolving_Bal         0  
Avg_Open_To_Buy             0  
Total_Amt_Chng_Q4_Q1        0  
Total_Trans_Amt             0  
Total_Trans_Ct              0  
Total_Ct_Chng_Q4_Q1         0  
Avg_Utilization_Ratio       0  
dtype: int64 (012)  
Test eda Passed (002)  
Test encoder_helper Passed (012)  
Test feature_engineering Passed (017)  
INFO:churn_predictor_class:Training Begining (015)  
INFO:churn_predictor_class:random forest results  
> test results
>               precision    recall  f1-score   support
> 
>            0       0.96      0.99      0.98      2543
>            1       0.93      0.80      0.86       496
> 
>     accuracy                           0.96      3039
>    macro avg       0.95      0.90      0.92      3039
> weighted avg       0.96      0.96      0.96      3039
> 
> train results
>               precision    recall  f1-score   support
> 
>            0       1.00      1.00      1.00      5957
>            1       1.00      1.00      1.00      1131
> 
>     accuracy                           1.00      7088
>    macro avg       1.00      1.00      1.00      7088
> weighted avg       1.00      1.00      1.00      7088
> 
> logistic regression results
> test results
>               precision    recall  f1-score   support
> 
>            0       0.90      0.97      0.93      2543
>            1       0.72      0.44      0.54       496
> 
>     accuracy                           0.88      3039
>    macro avg       0.81      0.70      0.74      3039
> weighted avg       0.87      0.88      0.87      3039
> 
> train results
>               precision    recall  f1-score   support
> 
>            0       0.91      0.96      0.94      5957
>            1       0.73      0.50      0.59      1131
> 
>     accuracy                           0.89      7088
>    macro avg       0.82      0.73      0.76      7088
> weighted avg       0.88      0.89      0.88      7088
> 
 (016)  
INFO:churn_predictor_class:Training Finished (017)  
Test train_model Passed (002)  
INFO:churn_predictor_class:Classification Report Started (018)  
INFO:churn_predictor_class:Classification Report Finished (024)  
Test classification_report_images Passed (002)  
INFO:churn_predictor_class:Features Importance Report Started (025)  
INFO:churn_predictor_class:Features Importance Report Finished (026)  
Test feature_importance_plot Passed (002)  
Tests Finished (019)  
INFO:churn_predictor_class:Tests Finished (019)  




Deploy some (from https://dejavu-fonts.github.io/ for example)

C:\Users\jbolivarg\.conda\envs\py38\Library\lib\fonts


