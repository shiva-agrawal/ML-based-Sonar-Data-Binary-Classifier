'''
 Author      : Shiva Agrawal
 Date        : 06.09.2018
 Version     : 1.0
 Description : test file for BinaryClassificationModelDevelopment
'''

import BinaryClassificationModelDevelopment as classification

if __name__ == '__main__':

    dataset = 'sonar.all-data.csv'
    classification.binaryClassificationModel(dataset)