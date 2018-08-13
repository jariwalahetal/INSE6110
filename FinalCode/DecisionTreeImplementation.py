from sklearn import  tree

class DecisionTree:
	def __init__(self,traindata, trainresult, testdata):
		self.traindata = traindata
		self.testdata = testdata
		self.trainresult = trainresult
		self.predictions = []

	def getPrediction(self):
		clf = tree.DecisionTreeClassifier()
		clf.fit(self.traindata,self.trainresult)
		self.predictions = clf.predict(self.testdata)
		return self.predictions