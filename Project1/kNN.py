# Name: Shagun Paul
# UTA ID: 1001557958

import math
import operator
import random

train_data = []

def open_file():
	return processed_data(file = open('ATNT50/trainDataXY.txt'))

def processed_data(file):
	lines = []
	raw_data = []
	data = []
	for index, line in enumerate(file):
		if index != 0:
			raw_data.append(line.rstrip().rsplit(','))
		else:
			lines = line.rstrip().rsplit(',')
	for x in range(0, len(lines)):
		index_list = [sample[x] for sample in raw_data]
		data.append(index_list + [lines[x]])
	return data

def getknnFit(train_data, test_data, k):
	result_test = [getNeighbors(train_data, testInstance, k) for testInstance in test_data]
	metric_test = [[item[-1] for item in neighbor] for neighbor in result_test]
	posn_count_test = 0
	for index, item in enumerate(test_data):
		if item[-1] == mode(metric_test[index]):
			posn_count_test+=1
	return (((posn_count_test/len(metric_test))*100))

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

def train_test_split(data):
	split_randInt = random.randint(int(len(data)/2), (len(data)-5))
	random.shuffle(data)
	train_data = data[0:split_randInt]
	test_data = data[split_randInt:-1]
	return train_data, test_data

def getEuclideanDistance(instance1, instance2, length):
	dist = 0
	for x in range(length):
		dist += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(dist)

def getNeighbors(trainSet, testInstance, k):
	distance = []
	length = len(testInstance)-1
	for x in range(len(trainSet)):
		dist = getEuclideanDistance(testInstance, trainSet[x], length)
		distance.append((trainSet[x], dist))
	distance.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distance[x][0])
	return neighbors

def main():
	data = open_file()
	train_data, test_data = train_test_split(data)
	for k in range(1,11):
		getknnFit(train_data, test_data, k)

if __name__ == '__main__':
	main()