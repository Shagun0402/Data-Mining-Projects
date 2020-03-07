# Name: Shagun Paul
# UTA ID:1001557958

import random
import pandas as pd
import numpy as np

def open_file(filename):
	return processed_data(file = open(filename))

def processed_data(file):
	lines = []
	rawData = []
	data = []
	for index, line in enumerate(file):
		if index != 0:
			rawData.append(line.rstrip().rsplit(','))
		else:
			lines = line.rstrip().rsplit(',')
	for x in range(0, len(lines)):
		index_list = [sample[x] for sample in rawData]
		data.append(index_list + [lines[x]])
	return data, lines

def split_train_test(data):
	split_randomInt = random.randint(int(len(data)/2), (len(data)-5))
	random.shuffle(data)
	train_data = data[0:split_randomInt]
	test_data = data[split_randomInt:-1]
	return train_data, test_data

# sub-routine 1
def pickDataClass(filename, class_ids):
    dataset, class_indexes = open_file(filename)
    picked_data = []
    for i in list(class_ids):
        picked_data.extend([j for j in dataset if j[-1] == str(i)])
    return picked_data

 # sub-routine 2
def splitData2TestTrain(filename, number_per_class,  test_instances):
    test_inst_count = int(test_instances.rsplit(":")[-1])
    if filename == "ATNTFaceImages400.txt":
        classes = 40
    else:
        classes = 26
    samples = number_per_class
    training_instances = samples - test_inst_count
    data = pd.read_csv(filename, header=-1).as_matrix()
    firstnum, secondnum = data.shape
    data_X = np.transpose(data[1:, :])
    data_Y = np.transpose(data[0, :])
    train_Y, test_Y = [], []
    train_X = np.zeros((1, firstnum - 1))
    test_X = np.zeros((1, firstnum - 1))
    class_numbers = np.arange(1, classes+1)
    for i in class_numbers:
        j = i - 1
        train_X = np.vstack((train_X, data_X[(samples * (j)):(samples * j) + training_instances, :]))
        train_Y = np.hstack((train_Y, data_Y[(samples * (j)):(samples * j) + training_instances]))
        test_X = np.vstack((test_X, data_X[(samples * j) + training_instances:(samples * j) + samples, :]))
        test_Y = np.hstack((test_Y, data_Y[(samples * j) + training_instances:(samples * j) + samples]))
    train_X = train_X[1:, :]
    test_X = test_X[1:, :]
    return train_X, train_Y, test_X, test_Y

# sub-routine 3
def store_data(trainX, trainY, testX, testY):
    train_dict = {"trainX":trainX, "trainY":trainY}
    test_dict = {"testX":testX, "testY":testY}

    file = open("trainData.txt","w")
    file.write(str(train_dict))
    file.close()

    file = open("testFile.txt","w")
    file.write(str(test_dict))
    file.close()

def letter_2_digit_convert(sample):
    return ([ ord(item.upper())-64 for item in sample])
    # letter_2_digit_convert("APPLE")
