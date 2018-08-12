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
from INSE6110.FinalCode.NaiveBayesImplementation import *

#define global variables
folds = 2
repeats = 1
K = 5
simulateResults = SimulateResults()
label = [0, 1]

# #step-1 generate preprocess data
print("Sep 1: making the columns header")
col_Names = ["Word_freq_make", "Word_freq_address",
             "Word_freq_all",
              "Word_freq_3d",
                 "Word_freq_our",
                 "Word_freq_over",
                 "Word_freq_remove",
                 "Word_freq_internet",
                 "Word_freq_order",
                 "Word_freq_mail",
                 "Word_freq_receive",
                 "Word_freq_will",
                 "Word_freq_people",
                 "Word_freq_report",
                 "Word_freq_addresses",
                 "Word_freq_free",
                 "Word_freq_business",
                 "Word_freq_email",
                 "Word_freq_you",
                 "Word_freq_credit",
                 "Word_freq_your",
                 "Word_freq_font",
                 "Word_freq_000",
                 "Word_freq_money",
                 "Word_freq_hp",
                 "Word_freq_hpl",
                 "Word_freq_george",
                 "Word_freq_650",
                 "Word_freq_lab",
                 "Word_freq_labs",
                 "Word_freq_telnet",
                 "Word_freq_857",
                 "Word_freq_data",
                 "Word_freq_415",
                 "Word_freq_85",
                 "Word_freq_technology",
                 "Word_freq_1999",
                 "Word_freq_parts",
                 "Word_freq_pm",
                 "Word_freq_direct",
                 "Word_freq_cs",
                 "Word_freq_meeting",
                 "Word_freq_original",
                 "Word_freq_project",
                 "Word_freq_re",
                 "Word_freq_edu",
                 "Word_freq_table",
                 "Word_freq_conference",
                 "Char_freq1",
                 "Char_freq2",
                 "Char_freq3",
                 "Char_freq4",
                 "Char_freq5",
                 "Char_freq6",
                 "Capital_run_length_average",
                 "Capital_run_length_longest",
                 "Capital_run_length_total",
                 "Spam"]

#step-2 generate data for spambase data and apply KNN
print("Step-2: generating dataset for spambase")
dataset = pd.read_csv("spambase.data",names=col_Names)
X = np.array(dataset.iloc[:, 0:48])
Y = np.array(dataset.iloc[:, -1])

#genrate k folds
kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=None)

#get predictions for each fold
print("processing k folds for spambase")
testRecords = []
predictedRecordsForKNN = []
predictedRecordsForNB = []

foldIndex = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    print(" generating prediction for folds no - " + str(foldIndex) + "..." )
    foldIndex = foldIndex + 1

    #GET KNN Prediction
    print("         Getting KNN results...")
    knn = KNN(K, X_train, Y_train,X_test)
    predictionsForKNNForTheFold = knn.getPredictions()

    # GET NB Prediction
    print("         Getting NB results...")
    nb = NB(X_train, Y_train, X_test)
    predictionsForNBForTheFold = nb.getPredictions()

    #appends records for displaying results
    print(Y_test)
    testRecords.append(Y_test)
    predictedRecordsForKNN.append(predictionsForKNNForTheFold)
    predictedRecordsForNB.append(predictionsForNBForTheFold)


#get confusionmatrix
cmForKNN = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForKNN)
print("Result for Spambase - KNN")
print(cmForKNN)

cmForNB = simulateResults.getConfusionMatrix(testRecords, predictedRecordsForNB)
print("Result for Spambase - NB")
print(cmForNB)

#plot confusionmatrix
simulateResults.plot_confusion_matrix(cmForKNN,label,'Confusion matrix for KNN - Spambase')
simulateResults.plot_confusion_matrix(cmForNB,label,'Confusion matrix for NB - Spambase')
















