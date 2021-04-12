
# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
import numpy as np

# load data
filename = 'Zeszyt1.csv'
namesToReadFromFile = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapaleniePecherzaMoczowego', 'zapalenieNerekPochodzeniaMiedniczkowego']

############## zapaleniePecherzaMoczowego ###############

print("\n =================== zapalenie pecherza moczowego ==================== \n ")

names1 = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapaleniePecherzaMoczowego']

dataframe = read_csv(filename, names=namesToReadFromFile)
array = dataframe.values

X1 = array[:,0:6]
Y1 = array[:,6]

(scores, pval) = chi2(X1, Y1)

scoresList1 = scores.tolist()
print(scoresList1)
rankingNumeric = []
allFeaturesNumber = len(scoresList1)
minimum = min(scoresList1)-1

for i in range(allFeaturesNumber):
    index = scoresList1.index(max(scoresList1))
    rankingNumeric.append(index)
    result = str(index + 1) + " " + names1[index]
    print(result)
    scoresList1[index] = minimum

############## zapalenieNerekPochodzeniaMiedniczkowego ###############

print("\n =================== zapalenieNerekPochodzeniaMiedniczkowego ==================== \n ")

names2 = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapalenieNerekPochodzeniaMiedniczkowego']

dataframe = read_csv(filename, names=namesToReadFromFile)
array = dataframe.values

X2 = array[:,0:6]
Y2 = array[:,7]

(scores, pval) = chi2(X2, Y2)

scoresList2 = scores.tolist()
print(scoresList2)
rankingNumeric = []
allFeaturesNumber = len(scoresList2)
minimum = min(scoresList2)-1

for i in range(allFeaturesNumber):
    index = scoresList2.index(max(scoresList2))
    rankingNumeric.append(index)
    result = str(index + 1) + " " + names2[index]
    scoresList2[index] = minimum
    print(result)

############## Suma ###############

print("\n =================== Suma ==================== \n ")

names3 = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapalenie...']

dataframe = read_csv(filename, names=namesToReadFromFile)
array = dataframe.values

X3 = array[:,0:6]

Y3_first_ilness = array[:,6]
Y3_second_ilness = array[:,7]
Y3 = Y3_second_ilness

for i in range(len(Y3_first_ilness)):
    Y3[i] = Y3_first_ilness[i] + (2 * Y3_second_ilness[i])

(scores, pval) = chi2(X3, Y3)

scoresList3 = scores.tolist()
print(scoresList3)
rankingNumeric = []
allFeaturesNumber = len(scoresList3)
minimum = min(scoresList3)-1

for i in range(allFeaturesNumber):
    index = scoresList3.index(max(scoresList3))
    rankingNumeric.append(index)
    result = str(index + 1) + " " + names3[index]
    scoresList3[index] = minimum
    print(result)

