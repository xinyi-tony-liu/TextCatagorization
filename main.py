import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from queue import PriorityQueue

def loadwords(file):
    r = []
    with open(file, 'r') as f:
        for line in f:
            r.append(line.strip())
        return r

_gwords = loadwords(r".\\datasets\\words.txt")
_gwordslen = len(_gwords)

def loadlabels(file):
    r = []
    r.append(0)
    with open(file, 'r') as f:
        for line in f:
            r.append(int(line.strip()))
        return r

testlabels = loadlabels(r".\\datasets\\testLabel.txt")
trainlabels = loadlabels(r".\\datasets\\trainLabel.txt")

testdata = [np.zeros(_gwordslen) for col in range(1501)]
traindata = [np.zeros(_gwordslen) for col in range(1501)]

def loaddata(vector, file):
    with open(file, 'r') as f:
        for line in f:
            vector[int(line.strip().split(' ')[0])][int(line.strip().split(' ')[1])] = 1

loaddata(testdata, r".\\datasets\\testData.txt")
loaddata(traindata, r".\\datasets.\\trainData.txt")

def I(E):
    if (len(E) == 0):
        return 1

    p1 = E.count(1) / len(E)
    p2 = E.count(2) / len(E)

    if(p1 != 0 and p2 != 0):
        return - p1 * math.log(p1, 2) - p2 * math.log(p2, 2)
    else:
        return 0

def AIG(E, E1, E2):
    return I(E) - (0.5 * I(E1) + 0.5 * I(E2))

def IGW(E, E1, E2):
    return I(E) - ((len(E1) -1) / len(E) * I(E1) + len(E2) / len(E) * I(E2))


class Node:
    def __init__(self, data, label, words, fn, w, wo):
        self.data = data
        self.label = label
        self.words = words
        self.w = w
        self.wo = wo
        self.estimate = 1
        if(label.count(1) < label.count(2)):
            self.estimate = 2

        maxweight = 0
        maxword = ""
        if(label.count(1) != 0 and label.count(2) != 0):
            for word in words:
                for file in range(1,1501):
                    nw = []
                    nwo = []
                    if(data[file][words.index(word)] == 1):
                        nw.append(label[file])
                    else:
                        nwo.append(label[file])
                weight = fn(label, nw, nwo)
                if(weight > maxweight):
                    maxweight = weight
                    maxword = word
            self.weight = maxweight
            self.word = maxword

    def __gt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight > other.weight

def dt():
    q = PriorityQueue()
    n = Node(traindata, trainlabels, _gwords, AIG, [], [])
    print(n.word, n.weight)

dt()