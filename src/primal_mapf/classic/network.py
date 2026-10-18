import torch
from ..types import NetworkOutput
import torch.nn as nn
import torch.nn.functional as F

class ACNet(nn.Module):
    def __init__(self, fov_size=10, a_size=5, rnn_size=512, goal_repr_size=12):
        super(ACNet, self).__init__()
        self.rnn_size = rnn_size


        self.conv1 = nn.Conv2d(4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv2 = nn.Conv2d(rnn_size // 4, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3 = nn.Conv2d(rnn_size // 2, rnn_size - goal_repr_size, kernel_size=2, stride=1, padding=0)


        self.goal_fc = nn.Linear(3, goal_repr_size)


        self.fc1 = nn.Linear(rnn_size, rnn_size)
        self.fc2 = nn.Linear(rnn_size, rnn_size)

        # LSTM
        self.lstm = nn.LSTMCell(rnn_size, rnn_size)


        self.policy_layer = nn.Linear(rnn_size, a_size)
        self.value_layer = nn.Linear(rnn_size, 1)
        self.blocking_layer = nn.Linear(rnn_size, 1)
        self.on_goal_layer = nn.Linear(rnn_size, 1)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.rnn_size, device=device),
                torch.zeros(batch_size, self.rnn_size, device=device))

    def forward(self, obs, goal, hx, cx):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)

        x = self.conv3(x)
        flat = F.relu(x.view(x.size(0), -1))

        g = F.relu(self.goal_fc(goal))
        hidden_input = torch.cat([flat, g], dim=1)

        h1 = F.relu(self.fc1(hidden_input))
        h2 = self.fc2(h1) # Linear

        # RESIDUAL CONNECTION:
        h3 = F.relu(h2 + hidden_input)

        hx, cx = self.lstm(h3, (hx, cx))

        policy_logits = self.policy_layer(hx)
        policy = F.softmax(policy_logits, dim=-1)
        policy_sig = torch.sigmoid(policy_logits)

        value = self.value_layer(hx)
        blocking = torch.sigmoid(self.blocking_layer(hx))
        on_goal = torch.sigmoid(self.on_goal_layer(hx))

        return NetworkOutput(policy, value, hx, cx, blocking, on_goal, policy_sig)
