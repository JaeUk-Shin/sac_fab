import torch.nn as nn
import torch.nn.functional as F


class SACActor(nn.Module):
    def __init__(self, dimS, nA, hidden1, hidden2):
        super(SACActor, self).__init__()
        self.nA = nA
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nA)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        p = F.softmax(self.fc3(x), dim=-1)

        return p
