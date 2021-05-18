'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Binary Classification  model development using machine learning algorithm for Sonar metal vs rocks data.
			   The model is used to predict whether the detected metal by the sonar is metal or rock dpending on
			   measured singals.
'''

'''
Dataset Information:

Name: Sonar metal vs rocks
Samples: 208
Features: 60 (all numeric)
output: 1 (M or R) - (Metal or Rock)

In this dataset, as all the input features are just signal measurements, they do not given some specific name.
So there is no header used specifically.

Hence for the column, only numbers define them. Hence from column 1 to 60, inout features
                                                           column 61, output class
                                                           
As there are only two outcomes, this is binery classification problem

'''


import pandas as pd
import matplotlib.pyplot as pyplt
import numpy as np
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC

import pickle


def binaryClassificationModel(CsvFileName):

    # step 1: Import the Dataset from CSV to python
    #--------------------------------------------------

    ML_data = pd.read_csv(CsvFileName, header = None)
    print(ML_data)


    # step 2: Separate input and output data for ML
    #---------------------------------------------------

    ML_data_array = ML_data.values
    ML_data_input = ML_data_array[:, 0:60]  # all rows and columns from index 0 to 59 (all 60 input features)
    ML_data_output = ML_data_array[:, 60]  # all rows and column index 60 (last column - Class (output))


    # step 3: Desciptive analysis of the dataset
    #---------------------------------------------------
    print(ML_data.shape)  # dimensions of the dataset (rows, cols)
    print(ML_data.dtypes) # dataypes of all the features and outcome
    print(ML_data.head(20))  # print first 20 samples (just for knowing the data)

    dataStatistics = ML_data.describe() # find mean, std dev, min value, max value, 25th, 50th, 75th percentile of each feature
    print(dataStatistics)
    pd.set_option('precision', 2)
    print(ML_data.corr(method = 'pearson')) # correlation matrix for all the features

    print(ML_data.groupby(60).size())  # total count of samples availble from each class (column index 60)

    # step 4: Data anaylysis using different plots
    # ---------------------------------------------------
    ML_data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)  # histogram plots
    ML_data.plot(kind='density', subplots=True, layout=(8, 8), sharex=False, sharey=False)  # density plots

    # correlation matrix
    fig = pyplt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ML_data.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)



    # step 5: Data preprocessing - Standardize the input features
    # ------------------------------------------------------------

    scalar = StandardScaler().fit(ML_data_input)
    ML_data_input = scalar.transform(ML_data_input)

    # step 6: separate train (70 %) and validation (30 %) dataset
    # -----------------------------------------------------------
    validation_size = 0.3
    seed = 7
    [X_train, X_validation, Y_train, Y_validation] = train_test_split(ML_data_input, ML_data_output,
                                                                      test_size=validation_size, random_state=seed)

    # step 7: As at this moment, it is difficult to predict the right choice of algorithms, multiple algorithms are
    #         are used to develop different models and then all are compared
    # 10 fold cross validation and classification accuracy metric is selected

    # six different algorithms tried for binary classification

    print('\n')
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('NB', GaussianNB()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))

    # evaluate each model
    all_cv_results = []
    all_names = []
    k_folds = 10

    for name, model in models:
        kfold = KFold(n_splits=k_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        all_cv_results.append(cv_results)
        all_names.append(name)
        print(name + ' : ' + str(cv_results.mean()) + ' (' + str(cv_results.std()) + ')')

    # Compare Algorithms
    fig = pyplt.figure()
    fig.suptitle(' Algorithm Comparison ')
    ax = fig.add_subplot(111)
    pyplt.boxplot(all_cv_results)
    ax.set_xticklabels(all_names)

    '''
    LR : 0.7923809523809525 (0.07940253084607508)
    LDA : 0.7571428571428571 (0.10958590372299067)
    NB : 0.667142857142857 (0.10379313981673294)
    KNN : 0.7785714285714286 (0.0777299820749927)
    CART : 0.6838095238095239 (0.10946582315914544)
    SVM : 0.8266666666666665 (0.07184639700016114)
    
    From the above results, SVM and LR are good fits 
    '''

    # Step 8: Validating the two selected models with validation dataset to find best fit
    #-----------------------------------------------------------------------------------

    # LR

    print('-----LR model validation--------')
    lr_model_tuple = models[0]
    lr_model = lr_model_tuple[1]
    print(lr_model)
    lr_model.fit(X_train, Y_train)
    lr_predictions = lr_model.predict(X_validation)
    print(accuracy_score(Y_validation, lr_predictions))
    print(confusion_matrix(Y_validation, lr_predictions))
    print(classification_report(Y_validation, lr_predictions))

    '''
    LR model validation Output:
    Accuracy: 0.746031746031746
    
    Confusion matrix:
    [[27  8]
    [ 8 20]]
    
    Classification report
    
             precision    recall  f1-score   support

          M       0.77      0.77      0.77        35
          R       0.71      0.71      0.71        28

    avg / total       0.75      0.75      0.75        63
    
    '''

    # SVM

    print('-----SVM model validation--------')
    svm_model_tuple = models[5]
    svm_model = svm_model_tuple[1]
    print(svm_model)
    svm_model.fit(X_train, Y_train)
    svm_predictions = svm_model.predict(X_validation)
    print(accuracy_score(Y_validation, svm_predictions))
    print(confusion_matrix(Y_validation, svm_predictions))
    print(classification_report(Y_validation, svm_predictions))

    '''
    SVM model validation output:
    Accuracy: 0.8412698412698413
    
    confusion matrix:
    [[33  2]
    [ 8 20]]
    
    Classification report:
    
             precision    recall  f1-score   support

          M       0.80      0.94      0.87        35
          R       0.91      0.71      0.80        28

    avg / total       0.85      0.84      0.84        63

    
    '''

    '''
    Hence SVM models fits best for the given dataset
    '''

    # step 9: save the final model (SVM   algorithm) using pickel package

    model_filename = 'SVM_model.sav'
    pickle.dump(svm_model, open(model_filename, 'wb'))

	
	# to display the plots
    pyplt.show()

