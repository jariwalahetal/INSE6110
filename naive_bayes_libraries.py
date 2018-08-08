import csv
import random
import math
import operator
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
file = r'spambase.data'
col_Names = ["Word_freq_make",
					"Word_freq_address",
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
dataset = pd.read_csv(file, names=col_Names)
print(dataset.head())
#(dataset.iloc[:,-1]).astype(int)
X = np.array(dataset.iloc[:, 0:48])
target = np.array(dataset.iloc[:, -1])
X_train, X_test, y_train,y_test = train_test_split(X,target,test_size=0.2, random_state=17)
print(X_train)
print(y_test)
X_train.astype(int)
gaussNb=GaussianNB()
gaussNb.fit(X_train,y_train)
print(gaussNb)
y_expect=y_test
y_pred= gaussNb.predict(X_test)
print(accuracy_score(y_expect,y_pred))