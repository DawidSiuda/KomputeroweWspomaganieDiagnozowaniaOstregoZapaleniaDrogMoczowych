# import csv
# from numpy import *
# from xmlrpc.client import Boolean
#
# from sklearn.feature_selection import chi2
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.metrics import accuracy_score
# from scipy.stats import ttest_rel
# import numpy as np
#
# temperatured = []   # temperatura
# nausea = []         # nudnosci
# lumbarPain = []     # bol ledzwiowy
# urinePushing = []   # popuszczanie moczu
# micturitionPains = []   # Bóle związane z oddawaniem moczu
# burningOfUrethra = []   # Pieczenie cewki moczowej
#
# inflammationOfUrinaryBladder = [] # zapalenie pęcherza moczowego
# nephritisOfRenalPelvisOrigin = [] # Zapalenie nerek pochodzenia miedniczkowego
#
# size = 0
#
# with open('Zeszyt1.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         temperatured.append(float(row[0]))
#         nausea.append(int(row[1]))
#         lumbarPain.append(int(row[2]))
#         urinePushing.append(int(row[3]))
#         micturitionPains.append(int(row[4]))
#         burningOfUrethra.append(int(row[5]))
#
#         inflammationOfUrinaryBladder.append(int(row[6]))
#         nephritisOfRenalPelvisOrigin.append(int(row[7]))
#
#         size += 1
#
#
# temperatured2 = temperatured.copy()
# temperatured2[1] = 99
#
# print(temperatured)
# print(temperatured2)
# print(nausea)


# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from sklearn.feature_selection import chi2
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# load data
filename = 'Zeszyt1.csv'
namesToReadFromFile = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapaleniePecherzaMoczowego', 'zapalenieNerekPochodzeniaMiedniczkowego']
names = ['temperatura', 'nudnosci', 'bolLedzwiowy', 'popuszczanieMoczu', 'boleZwiazaneZOddawaniemMoczu', 'pieczenieCewkiMoczowej', 'zapaleniePecherzaMoczowego']
dataframe = read_csv(filename, names=namesToReadFromFile)
array = dataframe.values

X = array[:,0:6]
Y = array[:,6]

(scores, pval) = chi2(X, Y)
scoresList = scores.tolist()
print(scoresList)
rankingNumeric = []
allFeaturesNumber = len(scoresList)
minimum = min(scoresList)-1


file = open("ranking.txt", "w")
for i in range(allFeaturesNumber):
    index = scoresList.index(max(scoresList))
    rankingNumeric.append(index)
    result = str(index + 1) + " " + names[index]
    print(result)
    file.write(result + "\n")
    scoresList[index] = minimum
file.close()
`
print("=======================================")
print(scoresList)
