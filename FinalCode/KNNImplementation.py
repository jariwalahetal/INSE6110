import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import math
import operator

class KNN:
    def __init__(self, k, traindata,trainresult, testdata):
        self.traindata = traindata
        self.testdata = testdata
        self.trainresult = trainresult
        self.k = k
        self.predictions = []

    def euclideanDistance(self,instance1, instance2):
        distance = 0.0
        for x in range(len(instance2)):
            distance = distance + float(pow((instance1[x] - instance2[x]), 2))
        return math.sqrt(distance)

    def getNeighbors(self,trainingSet, trainResult, testInstance, k):
        distances = []
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x])
            distances.append((trainingSet[x], trainResult[x], dist))
        distances.sort(key=operator.itemgetter(2))
        neighbors2 = []
        for x in range(k):
            neighbors2.append((distances[x][0], distances[x][1]))
        return neighbors2

    def getResponse(self,neighbors1):
        classVotes = {}
        for x in range(len(neighbors1)):
            response = neighbors1[x][1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getPredictions(self):
        self.predictions = []
        for x in range(len(self.testdata)):
            neighbors1 = self.getNeighbors(self.traindata, self.trainresult, self.testdata[x], self.k)
            result = self.getResponse(neighbors1)
            self.predictions.append(result)
        return  self.predictions

