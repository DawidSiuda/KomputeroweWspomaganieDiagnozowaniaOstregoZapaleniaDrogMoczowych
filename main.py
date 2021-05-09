
# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.table import Table
import pandas

def checkerboard_table(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):

        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]
        if val == 1:
            tb.add_cell(i, j, width, height, text=fmt.format(val),
                loc='center', facecolor="green")
        elif val == 0:
            tb.add_cell(i, j, width, height, text=fmt.format(val),
                        loc='center', facecolor="white")
        elif val == -1:
            tb.add_cell(i, j, width, height, text=fmt.format(val),
                        loc='center', facecolor="red")
    # Row Labels...
    for i, label in enumerate(data.columns):
        tb.add_cell(i, -1, width, height, text=label, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig

# Set stdOut on file
now = datetime.now()
dateAsFilename = now.strftime("%Y_%m_%d_%H.%M.%S.txt")
original_stdout = sys.stdout # Save a reference to the original standard output

logFile = open(dateAsFilename, "w+")
# sys.stdout = logFile # Change the standard output to the file we created.
# sys.stdout = original_stdout # Reset the standard output to its original value

# load data
filename = 'Zeszyt1.csv'
namesToReadFromFile = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapaleniePecherzaMoczowego', 'zapalenieNerekPochodzeniaMiedniczkowego']

######################################
# Create ranking
######################################

dataframe = read_csv(filename, names=namesToReadFromFile)
array = dataframe.values

X = array[:,0:6]
Y_first_ilness = array[:,6]
Y_second_ilness = array[:,7]
Y = [0] * len(Y_second_ilness)

for i in range(len(Y_first_ilness)):
    if (Y_first_ilness[i] == 1) or (Y_second_ilness[i] == 1):
        Y[i] = 1
    else:
        Y[i] = 0
    Y[i] = Y_first_ilness[i]

print("==================================")
print("== Sick/healthy:")
print("==================================")
print(Y)

print("==================================")
print("== Feature ranking:")
print("==================================")
(scores, pval) = chi2(X, Y)
scoresList = scores.tolist()
print(scoresList)
rankingNumeric = []
numberOfFeatures = len(scoresList)
minimum = min(scoresList)-1


for i in range(numberOfFeatures):
    index = scoresList.index(max(scoresList))
    rankingNumeric.append(index)
    result = str(index + 1) + " " + namesToReadFromFile[index]
    scoresList[index] = minimum
    print(result)


######################################
# Tests
######################################

# prepare for experiment
momentum1 = 0
momentum2 = .9
momentums = [momentum1, momentum2]
layerSize1 = 50
layerSize2 = 100
layerSize3 = 150
layerSizes = [layerSize1, layerSize2, layerSize3]
numberOfEstimators = len(layerSizes)*len(momentums)
numberOfObjects = len(X)
numberOfRepeats = 5
numberOfFolds = 2
rskf = RepeatedStratifiedKFold(n_splits=numberOfFolds, n_repeats=numberOfRepeats, random_state=12345) # Może użyć tu zwykłej walidacji krzyżowej?

# create three-dimensional arary tp store scores
# scores array 6x6 -> numberOfEstimators x numberOfFeatures
scores = [[[] for x in range(numberOfEstimators)] for y in range(numberOfFeatures)]
scoresAverage = [[] for y in range(numberOfEstimators)]
Y = np.array(Y) # convert to numpy array



# check for every number of features - top_1, top_2, ..., top_numberOfFeatures
for currentComputingNumberOfFeatures in range(1, numberOfFeatures+1):
    # Create currentX[numberOfObjects][]
    currentX = [[] for i in range(numberOfObjects)]

    # Extract currently computing features from array of all features
    for i in range(currentComputingNumberOfFeatures):
        featureIndex = rankingNumeric[i] # get index of next feature in ranking
        for j in range(numberOfObjects):
            currentX[j].append(X[j][featureIndex]) # append next feature in ranking to currentX for every object
    currentX = np.array(currentX) # convert to numpy array

    # for every fold
    for train, test in rskf.split(currentX, Y):

        X_train = currentX[train]
        X_test = currentX[test]

        y_train = Y[train]
        y_test = Y[test]

        estimatorIndex = 0
        for momentum in momentums:
            for layerSize in layerSizes:
                # prepare classifier
                classifier = MLPClassifier(solver='sgd',
                                           hidden_layer_sizes=layerSize,
                                           momentum=momentum,
                                           max_iter=7000,
                                           learning_rate_init=0.0001,
                                           nesterovs_momentum=True,
                                           n_iter_no_change=50,
                                           verbose=False,
                                           learning_rate='adaptive').fit(X_train, y_train)

                # predict with classifier
                y_pred = classifier.predict(X_test)

                # assess accuracy
                result = accuracy_score(y_test, y_pred) # result is a scalar
                scores[estimatorIndex][currentComputingNumberOfFeatures-1].append(result)
                estimatorIndex += 1

print("==================================")
print("== Scores:")
print("==================================")
print(scores)

# calculate and save averages for folds
print("==================================")
print("== Average for every folds")
print("==================================")
print("Momentum, layerSize, featuresNumber, average")

namesOfEstimators  = [str(momentum1) + " " + str(layerSize1), str(momentum1) + " " + str(layerSize2), str(momentum1) + " " + str(layerSize3),
    str(momentum2) + " " + str(layerSize1), str(momentum2) + " " + str(layerSize2), str(momentum2) + " " + str(layerSize3)]
for i in range(len(scores)):
    for j in range(len(scores[i])):
        if scores[i][j]:
            s = 0
            for score in scores[i][j]:
                s += score
            s /= numberOfRepeats*numberOfFolds
            scoresAverage[i].append(s)
            print(namesOfEstimators[i], " ", str(j+1), " ", str(s))

print(scoresAverage)

# get maximum results
maxResults = []
i = 0
for estimatorScores in scoresAverage:
    maximum = max(estimatorScores)
    index = estimatorScores.index(maximum)
    maxResults.append(scores[i][index])
    i += 1



# perform statistic tests and save results
print("==================================")
print("== Looking for the best set of parametars")
print("==================================")

namesOfEstimators  = ["[Momentum: " + str(momentum1) + ", LayerSize: " + str(layerSize1) + "]", "[Momentum: " +
    str(momentum1) + ", LayerSize: " + str(layerSize2) + "]", "[Momentum: " + str(momentum1) + ", LayerSize: " +
    str(layerSize3) + "]", "[Momentum: " + str(momentum2) + ", LayerSize: " + str(layerSize1) + "]",
    "[Momentum: " + str(momentum2) + ", LayerSize: " + str(layerSize2) + "]", "[Momentum: " + str(momentum2)
    + ", LayerSize: " + str(layerSize3) + "]"]

namesOfEstimatorsShort  = ["[M: " + str(momentum1) + ", L: " + str(layerSize1) + "]", "[M: " +
    str(momentum1) + ", L: " + str(layerSize2) + "]", "[M: " + str(momentum1) + ", L: " +
    str(layerSize3) + "]", "[M: " + str(momentum2) + ", L: " + str(layerSize1) + "]",
    "[M: " + str(momentum2) + ", L: " + str(layerSize2) + "]", "[M: " + str(momentum2)
    + ", L: " + str(layerSize3) + "]"]
alpha = .05

for i in range(len(maxResults)):
    if i == 0:
        print("Current the best solution: ", namesOfEstimators[i])
        idOfMax = i
        continue

    # print( "TEST DAW ", maxResults[i])
    test = ttest_rel(maxResults[i], maxResults[idOfMax])
    T = test.statistic
    p = test.pvalue

    if p > alpha:
        print("p: ", str(round(p, 4)), " T: ", str(round(T, 4)), " There are no significant statistical differences between ", namesOfEstimators[idOfMax], " and ", namesOfEstimators[i])
    elif T > 0:
        print("p: ", str(round(p, 4)), " T: ", str(round(T, 4)), " ", namesOfEstimators[i], " is better than ", namesOfEstimators[idOfMax])
        idOfMax = i

# mapOfResoults = [[0 for x in range(len(maxResults))] for y in range(len(maxResults))]

tabWidth = len(maxResults)
mapOfResoults = np.zeros(tabWidth * tabWidth)
for i in range(len(maxResults)):
    for j in range(i+1, len(maxResults)):


        # # print( "TEST DAW ", maxResults[i])
        # test = ttest_rel(maxResults[i], maxResults[idOfMax])
        # T = test.statistic
        # p = test.pvalue
        #
        # if p > alpha:
        #     print("p: ", str(round(p, 4)), " T: ", str(round(T, 4)), " There are no significant statistical differences between ", namesOfEstimators[idOfMax], " and ", namesOfEstimators[i])
        # elif T > 0:
        #     print("p: ", str(round(p, 4)), " T: ", str(round(T, 4)), " ", namesOfEstimators[i], " is better than ", namesOfEstimators[idOfMax])
        #     idOfMax = i

        test = ttest_rel(maxResults[i], maxResults[j])
        T = test.statistic
        p = test.pvalue
        # result = "p: " + str(p) + " T: " + str(T)
        # print(result)

        if p > alpha:
            mapOfResoults[i * tabWidth + j] = 0
            mapOfResoults[j * tabWidth + i] = 0
        elif T > 0:
            mapOfResoults[i * tabWidth + j] = 1
            mapOfResoults[j * tabWidth + i] = -1
        else:
            mapOfResoults[i * tabWidth + j] = -1
            mapOfResoults[j * tabWidth + i] = 1

print(mapOfResoults)

# Reshape things into a 9x9 grid.
mapOfResoults = mapOfResoults.reshape((tabWidth, tabWidth))

# row_labels = namesOfEstimators
# col_labels = namesOfEstimators
# plt.matshow(mapOfResoults)
# plt.xticks(range(tabWidth), col_labels)
# plt.yticks(range(tabWidth), row_labels)
#
# xcoords = [0.5, 1.5, 2.5, 3.5, 4.5]
# for xc in xcoords:
#     plt.axvline(x=xc)
#     # plt.ayvline(y=xc)
#
#
# plt.show()
#
# data = pandas.DataFrame(mapOfResoults,
#                         columns=namesOfEstimators)
# checkerboard_table(data)
# plt.show()

data = pandas.DataFrame(mapOfResoults,
                        columns=namesOfEstimatorsShort)
checkerboard_table(data)
plt.show()
# print("==================== DAWID ======================")
#
# for i in range(len(maxResults)):
#     for j in range(i+1, len(maxResults)):
#         test = ttest_rel(maxResults[i], maxResults[j])
#
#         print("sth dupa ", maxResults[i])
#
#         T = test.statistic
#         p = test.pvalue
#         result = "p: " + str(p) + " T: " + str(T)
#         print(result)
#
#         if p > alpha:
#             result = "Brak istotnych różnic statystycznych między " + namesOfEstimators[i] + " a " + namesOfEstimators[j] + "\n"
#         elif T > 0:
#             result = namesOfEstimators[i] + " osiągnął lepszy wynik od " + namesOfEstimators[j] + "\n"
#         else:
#             result = namesOfEstimators[j] + " osiągnął lepszy wynik od " + namesOfEstimators[i] + "\n"
#         print(result)


sys.stdout = original_stdout # Reset the standard output to its original value
logFile.close()