# Name: Shagun Paul
# UTA ID: 1001557958

import taskE_dataHandler
import linear_regression
import centroid_classifier
import kNN
import SVM


def splitData2TestTrain(filename, number_per_class,  test_instances):
    if isinstance(filename, str):
        dataset, class_indexes = open(filename)
    else:
        dataset = filename
        class_indexes = [item[-1] for item in dataset]

    unique_class_indexes = list(set(class_indexes))
    per_class_instances = []
    for i in unique_class_indexes:
        per_class_instances.append([j for j in dataset if str(j[-1]) == str(i)][:number_per_class])
    test_inst_count = int(test_instances.rsplit(":")[-1])
    trainX = []
    trainY = []
    testX = []
    testY = []
    for item in per_class_instances:
        testX.extend(item[:test_inst_count])
        trainX.extend(item[test_inst_count:])
    trainY = [item[-1] for item in trainX]
    testY = [item[-1] for item in testX]
    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = splitData2TestTrain(\
                                                taskE_dataHandler.pickDataClass(\
                                                   'HandwrittenLetters.txt',
                                                    taskE_dataHandler.letter_2_digit_convert(\
                                                               "abcde")),
                                                    39,
                                                    "1:9")


print("############################################################################")
print ("Centroid method: ")
acu_centroid = []
centroid_acc = centroid_classifier.predict(trainX, trainY, testX, testY, 4)
acu_centroid.append(centroid_acc)
print("Centroid Method Accuracy %s" %centroid_acc)

print("############################################################################")
print ("KNN method: ")
acu_knn = []
kn_acc = kNN.getknnFit(trainX, testX, 4)
acu_knn.append(kn_acc)
print ("KNN Method Accuracy %s" %kn_acc)

print("############################################################################")
print ("SVM method: ")
acu_svm = []
svm_acc = SVM.kfold_SVM(trainX, trainY, testX, testY)
acu_svm.append(svm_acc)
print("Accuracy for SVM Method %s" %(svm_acc * 100))

print("############################################################################")
print ("Linear Regression Method: ")
acu_lin_reg = []
result, lin_reg_acc = linear_regression.predict(trainX, trainY, testX, testY)
acu_lin_reg.append(lin_reg_acc)
print ("Linear Regression accuracy %s"%lin_reg_acc)
print("############################################################################")

