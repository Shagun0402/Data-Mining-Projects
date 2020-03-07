# Name: Shagun Paul
# UTA ID: 1001557958

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import taskE_dataHandler
import linear_regression
import centroid_classifier
import kNN
import SVM
import numpy as np
import random

def cross_validation(k, train_data, feature_names, classifier):
    for index, item in enumerate(train_data):
        item.append(feature_names[index])
    random.shuffle(train_data)
    k_splits = np.array_split(train_data, k)
    feature_splits = [[in_item[-1] for in_item in item]for item in k_splits]
    all_accuracy =  0
    for k in range(0,k):
        print ("For %s fold" %(int(k)+1))
        trainX = []
        trainY = []
        testX = k_splits[k]
        testY = feature_splits[k]
        trainX_temp = k_splits[:k] + k_splits[(k + 1):]
        trainY_temp = feature_splits[:k] + feature_splits[(k + 1):]
        for x in range(len(trainX_temp)):
            trainX.extend(trainX_temp[x])
            trainY.extend(trainY_temp[x])
        if classifier == 1:
            matrix, accuracy = (linear_regression.predict(trainX, trainY, testX, testY))
        elif classifier == 2:
            accuracy = (centroid_classifier.predict(trainX, trainY, testX, testY, 4))
        elif classifier == 3:
            accuracy = (kNN.getknnFit(trainX, testX, 4))
        print (abs(accuracy))
        all_accuracy += accuracy
    k_accuracy = float(all_accuracy)/5
    return abs(k_accuracy)

def getClassifierType(classifier):
    print("#######################################################################################")
    return {
        '1':"Linear Regression Classifier:",
        '2':"Centroid Classifier:",
        '3':"k- Nearest Neighbor Classifier:",
        '4':"Support vector Machine Classifier:"
    }[str(classifier)]

def driver(classifier):
    print (getClassifierType(classifier))
    if classifier == 4:
        trainX, trainY, testX, testY = taskE_dataHandler.splitData2TestTrain('ATNTFaceImages400.txt', 10, '1:8')
        print ("\n Average Accuracy for 5-fold Cross Validation is: %s"% SVM.kfold_SVM(trainX, trainY, testX, testY))
    else:
        data, indexes = taskE_dataHandler.open_file("ATNTFaceImages400.txt")
        print ("\nAverage Accuracy for 5-fold Cross Validation is: %s" % cross_validation(5, data, indexes, classifier))

def main():
    for i in range(1,5):
        driver(i)

if __name__ == '__main__':
    main()