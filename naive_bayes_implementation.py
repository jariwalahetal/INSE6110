from test_train_set_creation import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools
def separateByClass(xtrain, ytrain):
    len1 = len(xtrain)
    len2 = len(ytrain)
    separated = {}
    if (len1 == len2):
        for i in range(len1):
            classtype = ytrain[i]
            vector = xtrain[i]
            if (ytrain[i] not in separated):
                separated[classtype] = []
            separated[classtype].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataSet):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataSet)]
	return summaries


def summarizeByClass(xtrain, ytrain):
    separated = separateByClass(xtrain, ytrain)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities


def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions


def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == predictions[i]:
			correct += 1
	return (correct / float(len(testSet))) * 100.0

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
	accuracy = np.trace(cm) / float(np.sum(cm))
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	plt.show()

def main():
	readDataSet = readFile('spambase.data')
	xtrain, xtest, ytrain, ytest = splitDataIntoTrainTest(readDataSet, 0.2, 48)
	summarizedClass = summarizeByClass(xtrain, ytrain)
	print(summarizedClass)
	predictions = getPredictions(summarizedClass, xtest)
	accuracy = getAccuracy(ytest, predictions)
	print(('Accuracy: {0}%').format(accuracy))

	classification = classification_report(ytest, predictions)
	print(classification)
	label = [0, 1]
	cmatrix = confusion_matrix(ytest, predictions,label)
	print(cmatrix)
	plot_confusion_matrix(cmatrix,
						  label)
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#cax = ax.matshow(cmatrix)
	#plt.title('Confusion matrix of the classifier')
	#fig.colorbar(cax)
	#ax.set_xticklabels([''] + label)
	#ax.set_yticklabels([''] + label)
	#plt.tight_layout()
	#plt.xlabel('Predicted')
	#plt.ylabel('True')
	#plt.show()

main()
