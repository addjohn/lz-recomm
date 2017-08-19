import torch
import numpy as np
from Lib import util, net

print("Prediction Started")

testData, _ = util.loadData("Data/lezhin_dataset_v2_test_without_label.tsv", False, False)
print("Data Loaded")

plfS, buyS, schS, gnrS, usrS, etcS, varLenS = util.variablizeFeatures(testData, False)

# Load Parameters from Trained Model
model = net.Net(varLenS)
model.load_state_dict(torch.load("Model/trainedModel.pth"))

# Predict and Save Results to predictions.csv
prob = model.forward(plfS, buyS, schS, gnrS, usrS, etcS)
pred = [np.argmax(i) for i in prob.data.tolist()]

util.saveData("predictions.csv", pred)
print("Prediction Saved")
