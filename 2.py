import torch
from torch.utils.data import Dataset as set
# import sklearn.datasets
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearn.metrics import accuracy_score
from torch.autograd import Variable

X = np.array([[1],[2],[3],[4]], dtype=np.float32)
Y = np.array([[1],[5],[6],[7]], dtype=np.float32)
X = Variable(torch.from_numpy(X))
Y = Variable(torch.from_numpy(Y))


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def de_loss(output, target):
    return torch.abs(output - target)

model = MyClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 10000
losses = []
for i in range(epochs):
    for i in range(len(X)):
        inputs = X[i]
        targets = Y[i]
        #print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = de_loss(outputs.view([1,1]), (targets))
        print("loss, input, outputs, target")
        print(round(loss.data[0][0], 3), inputs.data[0], round(outputs.data[0],3), targets.data[0])
        loss.backward()
        optimizer.step()

