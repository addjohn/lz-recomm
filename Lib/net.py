import torch
import torch.nn as nn
import torch.nn.functional as ftn


class Net(nn.Module):

    def __init__(self, varLen):
        super(Net, self).__init__()
        self.plfDimI = varLen[0]
        self.plfDimH = 4
        self.plfDimO = 2

        self.buyDimI = varLen[1]
        self.buyDimO = 10

        self.schDimI = varLen[2]
        self.schDimO = 3

        self.gnrDimI = varLen[3]
        self.gnrDimH = 12
        self.gnrDimO = 6

        self.usrDimI = varLen[4]
        self.usrDim0 = 4

        self.etcDimI = varLen[5]

        self.plfLinI = nn.Linear(self.plfDimI, self.plfDimH)
        self.plfLinO = nn.Linear(self.plfDimH, self.plfDimO)
        self.buyEmb = nn.Linear(self.buyDimI, self.buyDimO)
        self.schEmb = nn.Linear(self.schDimI, self.schDimO)
        self.gnrLinI = nn.Linear(self.gnrDimI, self.gnrDimH)
        self.gnrLinO = nn.Linear(self.gnrDimH, self.gnrDimO)
        self.usrEmb = nn.Linear(self.usrDimI, self.usrDim0)

        self.prdDimI = self.plfDimO + self.buyDimO + self.schDimO + self.gnrDimO + self.usrDim0 + self.etcDimI
        self.prdDimH1 = int(self.prdDimI / 2.0 + 0.5)
        self.prdDimH2 = int(self.prdDimH1 / 2.0 + 0.5)
        self.prdDimO = 2
        self.prdLinI = nn.Linear(self.prdDimI, self.prdDimH1)
        self.prdLinH = nn.Linear(self.prdDimH1, self.prdDimH2)
        self.prdLinO = nn.Linear(self.prdDimH2, self.prdDimO)

    def plfForward(self, x):
        x = ftn.relu(self.plfLinI(x))
        x = ftn.relu(self.plfLinO(x))
        return x

    def buyForward(self, x):
        x = self.buyEmb(x)
        return x

    def schForward(self, x):
        x = self.schEmb(x)
        return x

    def gnrForwad(self, x):
        x = ftn.relu(self.gnrLinI(x))
        x = ftn.relu(self.gnrLinO(x))
        return x

    def usrForward(self, x):
        x = self.usrEmb(x)
        return x

    def forward(self, plf, buy, sch, gnr, usr, etc):

        plf = self.plfForward(plf)
        buy = self.buyForward(buy)
        sch = self.schForward(sch)
        gnr = self.gnrForwad(gnr)
        usr = self.usrForward(usr)

        x = torch.cat([plf, buy, sch, gnr, usr, etc], dim=1)
        x = ftn.relu(self.prdLinI(x))
        x = ftn.relu(self.prdLinH(x))
        x = ftn.softmax(self.prdLinO(x))
        return x