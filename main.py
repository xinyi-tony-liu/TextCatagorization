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

testlabel = loadlabels(r".\\datasets\\testLabel.txt")
trainlabel = loadlabels(r".\\datasets\\trainLabel.txt")

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

def WIG(E, E1, E2):
    return I(E) - ((len(E1) -1) / len(E) * I(E1) + len(E2) / len(E) * I(E2))


class Node:
    def __init__(self, data, label, words, fn):
        self.data = data
        self.label = label
        self.words = words
        self.fn = fn
        self.end = 0
        self.word = ""
        self.weight = 0
        self.estimate = 1
        self.ychild = None
        self.nchild = None
        if(label.count(1) < label.count(2)):
            self.estimate = 2

        maxweight = 0
        maxword = ""
        if(label.count(1) != 0 and label.count(2) != 0):
            for word in words:
                nw = []
                nwo = []
                pos = _gwords.index(word)
                l = len(label)
                for file in range(l):
                    if(data[file][pos] == 1):
                        nw.append(label[file])
                    else:
                        nwo.append(label[file])
                weight = fn(label, nw, nwo)
                if(weight > maxweight):
                    maxweight = weight
                    maxword = word
            self.weight = maxweight
            self.word = maxword
        else:
            self.end = 1
            
    def split(self):
        pos = _gwords.index(self.word)
        l = len(self.label)
        words = self.words.copy()
        del words[words.index(self.word)]
        n1data = []
        n2data = []
        n1label = []
        n2label = []
        for file in range (l):
            if (self.data[file][pos] == 1):
                n1data.append(self.data[file])
                n1label.append(self.label[file])
            else:
                n2data.append(self.data[file])
                n2label.append(self.label[file])
                
        n1 = Node(n1data,n1label,words,self.fn)
        n2 = Node(n2data,n2label,words,self.fn)
        self.ychild = n1
        self.nchild = n2
        return n1, n2
    
    def test(self, data, label):
        count = 0
        l = len(label)
        for file in range(l):
            node = self
            for i in range(100):
                if(node.end == 1):
                    break
                pos = _gwords.index(node.word)
                if (data[file][pos] == 1):
                    if(node.ychild == None):
                        break
                    node = node.ychild
                else:
                    if(node.nchild == None):
                        break
                    node = node.nchild
            if (node.estimate == label[file]):
                count = count + 1
        return count / l
            

    def __gt__(self, other):
        return self.weight < other.weight

    def __eq__(self, other):
        if(other == None):
            return False
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight > other.weight

def dt(fn, fname):
    q = PriorityQueue()
    root = Node(traindata, trainlabel, _gwords, fn)
    q.put(root)
    y1 = []
    y2 = []
    for i in range(100):
        y1.append(root.test(traindata, trainlabel))
        y2.append(root.test(testdata, testlabel))
        n = q.get()
        print(i, n.word, n.weight, n.estimate)
        if(n.end == 1):
            break
        n1, n2 = n.split()
        q.put(n1)
        q.put(n2)
    x = range(100)
    plt.plot(x, y1, label = "train accuracy")
    plt.plot(x, y2, label = "test accuracy")
    plt.xlabel("number of nodes")
    plt.ylabel("% correct")
    plt.title("Accuracy when using " + fname)
    plt.legend()
    plt.show()
    qcontent = []
    while(not q.empty()):
        qcontent.append(q.get().word)
    print(qcontent)

dt(AIG, "Average Information Gain")
dt(WIG, "Weighted Information Gain")