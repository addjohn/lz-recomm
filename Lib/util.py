import csv
import gzip
import numpy as np
import torch
from torch.autograd import Variable


def loadData(path, isZip=True, isTrain=True):
    data = []
    label = []
    if isZip:
        f = gzip.open(path, "rt")
    else:
        f = open(path, "r")
    readData = f.readlines()
    for row in readData:
        line = row.replace("\n", "").split("\t")
        if not isTrain:
            line = ['0'] + line
        for i, userFeature in enumerate(line[151:]):
            if userFeature == '':
                line[151 + i] = '0'
        data.append(line)
        label.append(int(line[0]))
    f.close()

    return np.array(data), label


def saveData(path, data):
    with open(path, "w") as f:
        writeData = csv.writer(f)
        for row in data:
            writeData.writerow([row])


def variablizeFeatures(x, grad=True):
    plf = np.array(x[:, 1:5]).astype(float)
    Nplf = plf.shape[1]
    Vplf = Variable(torch.FloatTensor(plf), requires_grad=grad)
    
    buy = np.array(x[:, 10:110]).astype(float)
    Nbuy = buy.shape[1]
    Vbuy = Variable(torch.FloatTensor(buy), requires_grad=grad)

    sch = np.array(x[:, 113:123]).astype(float)
    Nsch = sch.shape[1]
    Vsch = Variable(torch.FloatTensor(sch), requires_grad=grad)

    gnr = np.array(x[:, 123:141]).astype(float)
    Ngnr = gnr.shape[1]
    Vgnr = Variable(torch.FloatTensor(gnr), requires_grad=grad)

    usr = np.array(x[:, 151:]).astype(float)
    Nusr = usr.shape[1]
    Vusr = Variable(torch.FloatTensor(usr), requires_grad=grad)
    
    etc = np.array(x[:, [5, 8, 110, 111, 112, 141, 142, 143, 144]]).astype(float)
    Netc = etc.shape[1]
    Vetc = Variable(torch.FloatTensor(etc), requires_grad=grad)
    
    return Vplf, Vbuy, Vsch, Vgnr, Vusr, Vetc, [Nplf, Nbuy, Nsch, Ngnr, Nusr, Netc]
