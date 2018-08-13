import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import neighbors
import itertools
import math
import operator

class SimulateResults:
    def getAccuracy(self,testResult, predictions):
        correct = 0
        for x in range(len(testResult)):
            if testResult[x] == predictions[x]:
                correct += 1
        return (correct / float(len(testResult))) * 100

    def getConfusionMatrix(self,testResult, predictions):
        label = [0, 1]
        cmatrix = confusion_matrix(testResult, predictions, label)
        return cmatrix

    def plot_confusion_matrix(self,confusionmatrix,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=False):
        accuracy = np.trace(confusionmatrix) / float(np.sum(confusionmatrix))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]

        thresh = confusionmatrix.max() / 1.5 if normalize else confusionmatrix.max() / 2
        for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(confusionmatrix[i, j]),
                         horizontalalignment="center",
                         color="white" if confusionmatrix[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(confusionmatrix[i, j]),
                         horizontalalignment="center",
                         color="white" if confusionmatrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    def calculate(self,cmatrix):
        i = 0
        true_positive = cmatrix[i][0]
        false_negative = cmatrix[i][1]
        false_positive = cmatrix[i + 1][0]
        true_negative = cmatrix[i + 1][1]
        sdr = true_positive / (true_positive + false_negative)
        ldr = true_negative / (true_negative + false_positive)
        tdr = (true_positive + true_negative) / (true_positive + false_negative + true_positive + false_positive)
        return (sdr, ldr, tdr)

    def generateGraph(self,data):
        N = 4
        y = []
        z = []
        k = []
        width = 0.27
        ind = np.arange(N)
        for dict in data:
            classifier = dict['classifier']
            sdr = dict['sdr']
            ltr = dict['ltr']
            tdr = dict['tdr']
            y.append(sdr)
            k.append(tdr)
            z.append(ltr)

        ax = plt.subplot(111)
        rects1 = ax.bar(ind, y, width, color='b', align='center')
        rects2 = ax.bar(ind + width, z, width, color='g', align='center')
        rects3 = ax.bar(ind + width * 2, k, width, color='r', align='center')
        ax.legend((rects1[0], rects2[0], rects3[0]), ('SDR', 'LDR', 'TDR'))
        ax.set_ylabel('Detection Rate')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('KNN', 'Naive Bayes', 'SVM', 'Decision Trees'))

        plt.show()