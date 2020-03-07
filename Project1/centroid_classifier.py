# Name: Shagun Paul
# UTA ID: 1001557958
import math
import taskE_dataHandler
import operator
import random

train_data = []

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

def fetchCentroid(labelVectors):
	initial_centroid = []
	for j in range(len(labelVectors[0]) -1):
		sum = 0
		for i in range(len(labelVectors)):
			sum += int(labelVectors[i][j])
		sum = float(sum) / len(labelVectors)
		initial_centroid.append(sum)
	initial_centroid.append(labelVectors[-1][-1])
	return initial_centroid

def predict(trainX, trainY, testX, testY, k):

	uniqueTrainY = list(set(trainY))
	dict = {}
	tot_posn_count = 0

	for x in range (len(trainY)):
		dict.setdefault(trainY[x],[]).append(trainX[x])
	for testInstance in testX:
		train_data = [fetchCentroid(dict[key]) for key in dict.keys()]
		inst_result = getNeighbors(train_data, testInstance, k)
		class_predicted = mode([item[-1] for item in inst_result])
		if class_predicted == testInstance[-1]:
			tot_posn_count+=1
			dict.setdefault(class_predicted).append(testInstance)
	return (float(tot_posn_count)/len(testX))*100.00

def mode(numbers):
    large_count = 0
    modes = []
    for x in numbers:
        if x in modes:
            continue
        flag = numbers.count(x)
        if flag > large_count:
            del modes[:]
            modes.append(x)
            large_count = flag
        elif flag == large_count:
            modes.append(x)
    return modes[0]

def getEuclideanDistance(instance1, instance2, length):
	dist = 0
	for x in range(length):
		dist += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(dist)

def open_file():
	return processed_data(file = open('ATNT50/trainDataXY.txt'))

def getNeighbors(trainSet, testInstance, k):
	dist_arr = []
	length = len(testInstance) - 1
	for x in range(len(trainSet)):
		distance = getEuclideanDistance(testInstance, trainSet[x], length)
		dist_arr.append((trainSet[x], distance))
	dist_arr.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(dist_arr[x][0])
	return neighbors

def train_test_split(data):
	split_randInt = random.randint(int(len(data)/2), (len(data)-5))
	random.shuffle(data)
	train_data = data[0:split_randInt]
	test_data = data[split_randInt:-1]
	return train_data, test_data

def main(filename):
	# dataset = taskE_dataHandler.open_file(filename)
	trainX, trainY, testX, testY = taskE_dataHandler.splitData2TestTrain(taskE_dataHandler.pickDataClass('HandWrittenLetters.txt', taskE_dataHandler.letter_2_digit_convert("ABCDEFGHIJ")), 39, "1:20")
	predict(trainX, trainY, testX, testY, 4)

if __name__ == '__main__':
	filename = "ATNTFaceImages400.txt"
	main(filename)