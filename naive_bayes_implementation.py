from test_train_set_creation import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools


def separateByClass(xtrain, ytrain):
    len1 = len(ytrain)
    separated = {}
    for i in range(len1):
        classtype = ytrain[i]
        xdata = xtrain[i]
        if (ytrain[i] not in separated):
            separated[classtype] = []
        separated[classtype].append(xdata)
    return separated


def mean(numbers):
    ans = sum(numbers) / float(len(numbers))
    return ans


def standardDeviation(numbers):
    avg = mean(numbers)
    numerator = sum([pow(x - avg, 2) for x in numbers])
    denominator = float(len(numbers) - 1)
    variance = numerator / denominator
    return math.sqrt(variance)


def summarizeData(dataSet):
    summaries = [(mean(attribute), standardDeviation(attribute)) for attribute in zip(*dataSet)]
    return summaries


def summaryByClass(xtrain, ytrain):
    separated = separateByClass(xtrain, ytrain)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarizeData(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    val = (2 * math.pow(stdev, 2))
    exponent = math.exp(-(math.pow(x - mean, 2) / val))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, input):
    probabilities = {}
    for classValue, classSummary in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummary)):
            mean, stdev = classSummary[i]
            if stdev == 0:
                continue
            else:
                x = input[i]
                probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, input):
    probabilities = calculateClassProbabilities(summaries, input)
    bestClass, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestClass is None or probability > bestProb:
            bestProb = probability
            bestClass = classValue
    return bestClass


def getPredictions(summaries, xtest):
    predictions = []
    for i in range(len(xtest)):
        result = predict(summaries, xtest[i])
        predictions.append(result)
    return predictions


def getAccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest))) * 100.0


def plot_confusion_matrix(cmatrix,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    accuracy = np.trace(cmatrix) / float(np.sum(cmatrix))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]

    thresh = cmatrix.max() / 1.5 if normalize else cmatrix.max() / 2
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cmatrix[i, j]),
                     horizontalalignment="center",
                     color="white" if cmatrix[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cmatrix[i, j]),
                     horizontalalignment="center",
                     color="white" if cmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def main():
    while True:
        print("1. Display output for Spambase Dataset")
        print("2. Display output for SpamAssassin Dataset")
        print("3. Exit")
        try:
            selection = int(input("Select option to be performed: "))
            if selection <= 0 or selection > 3:
                continue
            elif selection == 1:
                readDataSet = readFile('spambase.data', )
                xtrain, xtest, ytrain, ytest = splitDataIntoTrainTest(readDataSet, 0.2, 'spambase.data')
                summarizedClass = summaryByClass(xtrain, ytrain)
                predictions = getPredictions(summarizedClass, xtest)
                accuracy = getAccuracy(ytest, predictions)
                print(('Accuracy: {0}%').format(accuracy))

                classification = classification_report(ytest, predictions)
                print(classification)
                label = [0, 1]
                cmatrix = confusion_matrix(ytest, predictions, label)
                print(cmatrix)
                plot_confusion_matrix(cmatrix,
                                      label)

            elif selection == 2:
                readDataSet = readFile('preprocess1.csv')
                xtrain, xtest, ytrain, ytest = splitDataIntoTrainTest(readDataSet, 0.2, 'preprocess1.csv')
                summarizedClass = summaryByClass(xtrain, ytrain)
                predictions = getPredictions(summarizedClass, xtest)
                accuracy = getAccuracy(ytest, predictions)
                print(('Accuracy: {0}%').format(accuracy))

                classification = classification_report(ytest, predictions)
                print(classification)
                label = [0, 1]
                cmatrix = confusion_matrix(ytest, predictions, label)
                print(cmatrix)
                plot_confusion_matrix(cmatrix,
                                      label)
            elif selection == 3:
                exit()
        except ValueError:
            print("please select from the options provided")


main()
