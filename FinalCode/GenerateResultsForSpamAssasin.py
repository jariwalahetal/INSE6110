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
from INSE6110.FinalCode.SimulateResults import *
from INSE6110.FinalCode.KNNImplementation import *
from INSE6110.FinalCode.NaiveBayesImplementation import *
from INSE6110.FinalCode.DecisionTreeImplementation import  *
from INSE6110.FinalCode.SVMImplementation import *

#define global variables
folds = 10
repeats = 1
K = 5
simulateResults = SimulateResults()
label = [0, 1]

# #step-1 generate preprocess data
print("Step 1:Generating CSV file...")
StartPreprocessing()
print("Preprocessing completed")

#step-2 generate data for spambase data and apply KNN
print("Step-2: generating dataset for spamAssasin")
dataset = pd.read_csv("FinalPreprocessDataForSpamAssassin.csv")
X = np.array(dataset.iloc[:, 0:9])
X.astype(float)
Y = np.array(dataset.iloc[:, -1])

#genrate k folds
kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=None)

#get predictions for each fold
print("processing k folds for spambase")
testRecords = []
predictedRecordsForKNN = []
predictedRecordsForNB = []
predictedRecordsForDT = []
predictedRecordsForSVM = []

foldIndex = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    print(" generating prediction for folds (no - " + str(foldIndex) + ")..." )
    foldIndex = foldIndex + 1

    #GET KNN Prediction
    print("         Getting KNN results...")
    knn = KNN(K, X_train, Y_train,X_test)
    predictionsForKNNForTheFold = knn.getPredictions()

    # GET NB Prediction
    print("         Getting NB results...")
    nb = NB(X_train, Y_train, X_test)
    predictionsForNBForTheFold = nb.getPredictions()

    # GET Decision Tree Prediction
    print("         Getting Decision Tree results...")
    dt = DecisionTree(X_train, Y_train, X_test)
    predictedRecordsForDTForTheFold = dt.getPrediction()

    # GET SVM Prediction
    print("         Getting SVM results...")
    svm = SupportVectorMachine(X_train, Y_train, X_test)
    predictedRecordsForSVMForTheFold = svm.getPrediction()

    #appends records for displaying results
    for t in Y_test:
        testRecords.append(t)

    for t in predictionsForKNNForTheFold:
        predictedRecordsForKNN.append(t)


    for t in predictionsForNBForTheFold:
        predictedRecordsForNB.append(t)

    for t in predictedRecordsForDTForTheFold:
        predictedRecordsForDT.append(t)

    for t in predictedRecordsForSVMForTheFold:
        predictedRecordsForSVM.append(t)


#get confusionmatrix
cmForKNN = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForKNN)
print("Result for Spambase - KNN")
print(cmForKNN)
print(simulateResults.getAccuracy(testRecords,predictedRecordsForKNN))


cmForNB = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForNB)
print("Result for SpamAssassin - NB")
print(cmForNB)
print(simulateResults.getAccuracy(testRecords,predictedRecordsForNB))

cmForDT = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForDT)
print("Result for SpamAssassin - Decision Tree")
print(cmForDT)
print(simulateResults.getAccuracy(testRecords,predictedRecordsForDT))

cmForSVM = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForSVM)
print("Result for SpamAssassin - SVM")
print(cmForSVM)
print(simulateResults.getAccuracy(testRecords,predictedRecordsForSVM))

#plot confusionmatrix
simulateResults.plot_confusion_matrix(cmForKNN,label,'Confusion matrix for KNN - SpamAssassin')
simulateResults.plot_confusion_matrix(cmForNB,label,'Confusion matrix for NB - SpamAssassin')
simulateResults.plot_confusion_matrix(cmForKNN,label,'Confusion matrix for Decision Tree - SpamAssassin')
simulateResults.plot_confusion_matrix(cmForNB,label,'Confusion matrix for SVM - SpamAssassin')

#Step 3: Preprocess Data
print("Preprocess Data")
processedData = []

sdr,ldr,tdr= simulateResults.calculate(cmForKNN)
processedData.append({"classifier":"KNN", "sdr":sdr, "ldr":ldr, "tdr":tdr})

sdr,ldr,tdr= simulateResults.calculate(cmForNB)
processedData.append({"classifier":"NB", "sdr":sdr, "ldr":ldr, "tdr":tdr})

sdr,ldr,tdr= simulateResults.calculate(cmForDT)
processedData.append({"classifier":"DT", "sdr":sdr, "ldr":ldr, "tdr":tdr})

sdr,ldr,tdr= simulateResults.calculate(cmForSVM)
processedData.append({"classifier":"SVM", "sdr":sdr, "ldr":ldr, "tdr":tdr})

print(processedData)

#Step 4: Show graph
simulateResults.generateGraph(processedData)
