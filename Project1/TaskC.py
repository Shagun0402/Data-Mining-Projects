# Name: Shagun Paul
# UTA ID: 1001557958

import taskE_dataHandler
import centroid_classifier
import matplotlib.pyplot as plt
arr = []

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

sample = "lwmuskcbtg"
split_desc = ["1:34", "1:29", "1:24", "1:19", "1:14", "1:9", "1:4"]
for split in split_desc:
    print ("Current split %s"%split)
    trainX, trainY, testX, testY = splitData2TestTrain(\
                                                        taskE_dataHandler.pickDataClass(\
                                                       'HandWrittenLetters.txt',
                                                        taskE_dataHandler.letter_2_digit_convert(\
                                                            sample)),
                                                        39,
                                                        split)


    arr.append(centroid_classifier.predict(trainX, trainY, testX, testY, 4))

print(arr)
plt.plot(arr)
plt.show()