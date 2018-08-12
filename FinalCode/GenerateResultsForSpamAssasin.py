import sys
from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
from INSE6110.FinalCode.PreprocessForSpamAssasinDatabase import StartPreprocessing
import numpy as np
sys.path.append("..")
from INSE6110.FinalCode.PreprocessForSpamAssasinDatabase import *
from INSE6110.FinalCode.SimulateResults import *
from INSE6110.FinalCode.KNNImplementation import *

#define global variables
folds = 10
repeats = 10
K = 5
simulateResults = SimulateResults()
label = [0, 1]

# #step-1 generate preprocess data
# doesn't require for this data set

#step-2 generate data for spambase data and apply KNN
print("Step-2: generating dataset for spambase\n")
dataset = pd.read_csv("FinalPreprocessData2.csv")
X = np.array(dataset.drop(['Spam'], 1))
Y = np.array(dataset['Spam'])

#genrate k folds
kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=None)

#get predictions for each fold
print("processing k folds for spambase")
testRecords = []
predictedRecords = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    print("generating prediction for folds \n")
    knn = KNN(K, X_train, Y_train,X_test)
    predictions = knn.getPredictions()

    #appends records for displaying results
    testRecords.append(Y_test)
    predictedRecords.append(predictions)

#get confusionmatrix
cm = simulateResults.getConfusionMatrix(testRecords)
print("Result for Spambase - KNN")
print(cm)

#plot confusionmatrix
simulateResults.plot_confusion_matrix(cm,label,'Confusion matrix for KNN - SpamAssasin')
