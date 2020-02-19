import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from queue import PriorityQueue

def csv2ll(file):
    with open(file) as f:
        return list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists

def loadtxt(file):
    r = []
    with open(file, 'r') as f:
        for line in f:
            r.append(line)
        return r #reads text into a list lines

class Node:
    def __init__():
        self.queue = []

def train():
    return 0

def test():
    return 0

def dt():
    testdata = loadtxt(r".\\datasets\\testData.txt")
    traindata = loadtxt(r".\\datasets.\\trainData.txt")
    testlabels = loadtxt(r".\\datasets\\testLabel.txt")
    trainlabels = loadtxt(r".\\datasets\\trainLabel.txt")
    words = loadtxt(r".\\datasets\\words.txt")


    return testdata[0]

b = dt()
print(b)