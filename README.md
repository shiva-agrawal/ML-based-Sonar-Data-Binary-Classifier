# Binary Classification Model development for Sonar measurement data

This project use the datset of Sonar mines vs rocks detection and develop the Machine learning predictive model (Binary classification model). It first import the dataset into Pandas Dataframe structure and then do the prepreprocessing and analysis of the data using scikit-learn library of Python.

After that, multiple ML models (using scikit-learn) are developed both linear and non linear for the same dataset with 30 % validation data and 70 % training data. All the developed models are then compared to find the best using Kfold cross validation and Classification  accuracy metric.

For this, it is found that LR (Linear Regression) and SVM (Support Vector Machine) classifier fits the best. Hence both these models are further investigated and finally SVM is selected for the model after validation.The model is then saved using the Pickle library of the Python for future use.

The project is implemented in Python (with Pycharm IDE) and Ubuntu 16.04 OS.

## Folder structure

1. docs
    * project details.pdf                                 - it is short project report
    * ML_modelDevelopmentTemplate.py                      - template used as reference to follow the steps for model development
2. results
    * SVM_model.sav                                       - developed model saved as .sav file
    * results.txt                                         - model accuracy results of the project copied to txt file
    * Plots for reference
         * Algorithm comparison with rescaling of features with accuracy metric 
         * Algorithm comparison with Standardization of features with accuracy metric 
         * Algorithm comparison without preprocessing of features and with accuracy metric 
         * correlation matrix of all 60 features
         * histogram plot of all features
         * pdf plot of all features
3. src
    * dataset (folder)                        - it contains the data and header of the Sonar mines vs rocks data as two csv files
    * BinaryClassificationModelDevelopment.py - documented and tested source code of the project
    * test.py                                 - file to do testing of the project
