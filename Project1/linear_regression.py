# Name: Shagun Paul
# UTA ID: 1001557958
import numpy as np

def predict(trainX, trainY, testX, testY):
    Xtrain=np.array(trainX,np.int32)
    Ytrain=np.array(trainY,np.int32)
    Xtest=np.array(testX,np.int32)
    Ytest=np.array(testY,np.int32)

    trainA = np.ones((len(trainX),len(trainX[0])))
    testA = np.ones((len(testX),len(testX[0])))

    padding_trainX = np.row_stack((Xtrain,trainA))
    padding_testX = np.row_stack((Xtest,testA))

    '''Computing Coefficients'''
    B_padding = np.dot(np.linalg.pinv(Xtrain), Ytrain.T)
    Ytest_padding = np.dot(B_padding.T,Xtest.T)
    Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
    err_test_padding = Ytest - Ytest_padding_argmax

    test_acc_padding = -(float((1-np.nonzero(err_test_padding)[0].size)/len(err_test_padding)))*100
    return Ytest_padding, test_acc_padding