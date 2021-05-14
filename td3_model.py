import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleCritic(nn.Module):

    def __init__(self, dimS, nA, hidden1, hidden2):
        super(DoubleCritic, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nA)

        self.fc4 = nn.Linear(dimS, hidden1)
        self.fc5 = nn.Linear(hidden1, hidden2)
        self.fc6 = nn.Linear(hidden2, nA)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(state))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

    def Q1(self, state):
        x = torch.cat(state, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x