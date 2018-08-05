import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator


# importing dataset
dataset = pd.read_csv("spambase.data")
X = np.array(dataset.drop(['Spam'], 1))
Y = np.array(dataset['Spam'])

# splitting data set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30
                                                    , random_state=0)


def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance2)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, trainResult, testInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], trainResult[x], dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0], distances[x][1]))
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
           classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, testResult, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testResult[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100


k = 5
predictions = []
for x in range(len(X_test)):
    neighbors = getNeighbors(X_train,Y_train, X_test[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(Y_test[x]))

accuracy = getAccuracy(X_test, Y_test, predictions)
print(accuracy)
