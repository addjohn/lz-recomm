import torch
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import KFold
from Lib import util, net

print("Cross Validation Started")

trainingData, label = util.loadData("Data/lezhin_dataset_v2_training.tsv.gz")
print("Data Loaded")

epoch = 1000
batch = 500
learningRate = 1e-3
earlyStopLevel = 1e-4
display = 10

# 5 - fold cross validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
validationError = 0
print("Fold Setting Finished")

fold = 1
for train, valid in kf.split(trainingData):

    plfV, buyV, schV, gnrV, usrV, etcV, varLenV = util.variablizeFeatures(trainingData[valid, :], False)

    labelV = [label[i] for i in valid]
    yv = Variable(torch.LongTensor(labelV), requires_grad=False)

    model = net.Net(varLenV)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    displayLoss = 1 / learningRate
    breakLimit = 3

    t = 1
    while 1:

        batchIdx = np.random.choice(train, batch, False)
        plfT, buyT, schT, gnrT, usrT, etcT, _ = util.variablizeFeatures(trainingData[batchIdx, :])

        labelT = [label[i] for i in batchIdx]
        yt = Variable(torch.LongTensor(labelT), requires_grad=False)

        ytPrd = model.forward(plfT, buyT, schT, gnrT, usrT, etcT)
        lossTrain = criterion(ytPrd, yt)

        if t % display == 1:
            yvPrd = model.forward(plfV, buyV, schV, gnrV, usrV, etcV)
            lossValid = criterion(yvPrd, yv)
            print("[%d - %d] loss : %.5f  ||  valid loss : %.8f" % (fold, t, lossTrain.data[0], lossValid.data[0]))
            if lossValid.data[0] - displayLoss > earlyStopLevel:
                breakLimit += -1
                if breakLimit == 0:
                    break
            displayLoss = lossValid.data[0]

        if t > epoch:
            yvPrd = model.forward(plfV, buyV, schV, gnrV, usrV, etcV)
            lossValid = criterion(yvPrd, yv)
            print("[%d - %d] loss : %.5f  ||  valid loss : %.8f" % (fold, t, lossTrain.data[0], lossValid.data[0]))
            break

        optimizer.zero_grad()
        lossTrain.backward()
        optimizer.step()

        t += 1

    hitCount = 0
    for i, prd in enumerate(yvPrd.data.tolist()):
        if np.argmax(np.array(prd)) == labelV[i]:
            hitCount += 1

    print("--")
    print("%d Fold Score : %.5f" % (fold, hitCount / len(valid)))
    print("--")

    validationError += lossValid.data[0]
    fold += 1

print("Validation Error : %.5f" % (validationError / k))
print("--")
print("Cross Validation Finished")
print("--")
