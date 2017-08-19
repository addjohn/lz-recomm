import torch
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
from Lib import util, net

print("Training Started")

trainingData, label = util.loadData("Data/lezhin_dataset_v2_training.tsv.gz")
print("Data Loaded")

epoch = 10000
batch = 500
learningRate = 1e-4
earlyStopLevel = 1e-4
display = 10

# Split Training : Test = 9 : 1
xTrain, xValid, yTrain, yValid = train_test_split(trainingData, label, test_size=0.1, random_state=42)
print("Split Setting Finished")

plfV, buyV, schV, gnrV, usrV, etcV, varLenV = util.variablizeFeatures(xValid, False)
yv = Variable(torch.LongTensor(yValid), requires_grad=False)

model = net.Net(varLenV)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

displayLoss = 1 / learningRate
breakLimit = 3

t = 1
while 1:

    batchIdx = np.random.choice(range(len(xTrain)), batch, False).astype(int)
    plfT, buyT, schT, gnrT, usrT, etcT, _ = util.variablizeFeatures(xTrain[batchIdx, :])

    yTrain = np.array(yTrain)
    yt = Variable(torch.LongTensor(yTrain[batchIdx]), requires_grad=False)

    ytPrd = model.forward(plfT, buyT, schT, gnrT, usrT, etcT)
    lossTrain = criterion(ytPrd, yt)

    if t % display == 1:
        yvPrd = model.forward(plfV, buyV, schV, gnrV, usrV, etcV)
        lossValid = criterion(yvPrd, yv)
        print("[%d] loss : %.5f  ||  valid loss : %.8f" % (t, lossTrain.data[0], lossValid.data[0]))
        torch.save(model.state_dict(), "Model/trainedModel.pth")
        if lossValid.data[0] - displayLoss > earlyStopLevel:
            breakLimit += -1
            if breakLimit == 0:
                break
        displayLoss = lossValid.data[0]

    if t > epoch:
        yvPrd = model.forward(plfV, buyV, schV, gnrV, usrV, etcV)
        lossValid = criterion(yvPrd, yv)
        print("[%d] loss : %.5f  ||  valid loss : %.8f" % (t, lossTrain.data[0], lossValid.data[0]))
        torch.save(model.state_dict(), "Model/trainedModel.pth")
        break

    optimizer.zero_grad()
    lossTrain.backward()
    optimizer.step()

    t += 1

hitCount = 0
for i, prd in enumerate(yvPrd.data.tolist()):
    if np.argmax(np.array(prd)) == yValid[i]:
        hitCount += 1

print("--")
print("Training Score : %.5f" % (hitCount / len(yValid)))
print("--")
print("Training Finished")
print("--")
