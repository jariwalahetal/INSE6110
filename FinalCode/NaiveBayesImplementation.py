import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools

class NB:
    def __init__(self, traindata,trainresult, testdata):
        self.traindata = traindata
        self.testdata = testdata
        self.trainresult = trainresult
        self.predictions = []

    def separateByClass(self):
        len1 = len(self.trainresult)
        separated = {}
        for i in range(len1):
            classtype = self.trainresult[i]
            xdata = self.traindata[i]
            if (self.trainresult[i] not in separated):
                separated[classtype] = []
            separated[classtype].append(xdata)
        return separated

    def mean(self,numbers):
        ans = sum(numbers) / float(len(numbers))
        return ans

    def standardDeviation(self,numbers):
        avg = self.mean(numbers)
        numerator = sum([pow(x - avg, 2) for x in numbers])
        denominator = float(len(numbers) - 1)
        variance = numerator / denominator
        return math.sqrt(variance)

    def summarizeData(self,dataSet):
        summaries = [(self.mean(attribute), self.standardDeviation(attribute)) for attribute in zip(*dataSet)]
        return summaries

    def summaryByClass(self):
        separated = self.separateByClass()
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarizeData(instances)
        return summaries

    def calculateClassProbabilities(self,summaries, input):
        probabilities = {}
        for classValue, classSummary in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummary)):
                mean, stdev = classSummary[i]
                if stdev == 0:
                    continue
                else:
                    x = input[i]
                    probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def calculateProbability(self,x, mean, stdev):
        val = (2 * math.pow(stdev, 2))
        exponent = math.exp(-(math.pow(x - mean, 2) / val))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict(self,summaries, input):
        probabilities = self.calculateClassProbabilities(summaries, input)
        bestClass, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestClass is None or probability > bestProb:
                bestProb = probability
                bestClass = classValue
        return bestClass

    def getPredictions(self):
        predictions = []
        summaries = self.summaryByClass()
        for i in range(len(self.testdata)):
            result = self.predict(summaries, self.testdata[i])
            predictions.append(result)
        return predictions