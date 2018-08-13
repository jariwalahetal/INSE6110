from setuptools.command.saveopts import saveopts
from sklearn import svm

class SupportVectorMachine:
	def __init__(self,traindata, trainresult, testdata):
		self.traindata = traindata
		self.testdata = testdata
		self.trainresult = trainresult
		self.predictions = []

	def getPrediction(self):
		params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
					   'gamma': [0.0001, 0.001, 0.01, 0.1],
					   'kernel': ['linear', 'rbf']}

		clf = svm.SVC(kernel='rbf', C=100, gamma=0.01)
		clf.fit(self.traindata, self.trainresult)
		self.predictions = clf.predict(self.testdata)
		return  self.predictions