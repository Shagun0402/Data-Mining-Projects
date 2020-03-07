# Name: Shagun Paul
# UTA ID: 1001557958
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn import svm
# import taskE_dataHandler
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import cross_val_score

label_encode = preprocessing.LabelEncoder

# trainX, trainY, testX, testY = data_handlerTE.splitData2TestTrain("ATNTFaceImages400.txt", 10, '1:6')
# kfold_SVM(trainX, trainY, testX, testY)

def eval_score(trainX, trainY, testX, testY):
    return predict(trainX, trainY, testX, testY).score

def predict(trainX, trainY, testX, testY):
    Xtrain=np.array(trainX)
    Ytrain=np.array(trainY)
    Xtest=np.array(testX)
    Ytest=np.array(testY)
    clf=svm.SVC()
    clf.fit(Xtrain,Ytrain)
    predicted= clf.predict(Xtest)
    print("Classification report for classifier %s:\n%s\n"
         % (clf, metrics.classification_report(Ytest, predicted)))
    return clf

def kfold_SVM(trainX, trainY, testX, testY):
    classify = predict(trainX, trainY, testX, testY)
    scores = cross_val_score(classify, testX, testY, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2))
    return scores.mean()
