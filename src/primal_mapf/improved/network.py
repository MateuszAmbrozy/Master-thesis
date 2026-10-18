import torch
from ..types import NetworkOutput
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        identity = x
        x = F.relu(self.block1(x))
        x = self.block2(x)
        x += identity
        return F.relu(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        batch_size, num_agents, input_dim = input.size()

        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1)

        scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (self.output_dim**0.5)
        scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads*self.output_dim)
        output = self.W_O(context)

        return output

class ACNet(nn.Module):
    def __init__(self, fov_size=10, a_size=5, rnn_size=512, goal_repr_size=12, num_heads=4):
        super(ACNet, self).__init__()
        self.rnn_size = rnn_size
        self.fov_size = fov_size

        self.conv1 = nn.Conv2d(4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(rnn_size // 4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(rnn_size // 4, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.res2 = ResBlock(rnn_size // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(rnn_size // 2, rnn_size - goal_repr_size, kernel_size=2, stride=1, padding=0)

        self.goal_fc = nn.Linear(3, goal_repr_size)

        self.fc1 = nn.Linear(rnn_size, rnn_size)
        self.fc2 = nn.Linear(rnn_size, rnn_size)

        self.attention = MultiHeadAttention(input_dim=rnn_size, output_dim=rnn_size, num_heads=num_heads)
        self.comm_mixer = nn.Linear(rnn_size * 2, rnn_size)

        self.lstm = nn.LSTMCell(rnn_size, rnn_size)

        self.policy_layer = nn.Linear(rnn_size, a_size)
        self.value_layer = nn.Linear(rnn_size, 1)
        self.blocking_layer = nn.Linear(rnn_size, 1)
        self.on_goal_layer = nn.Linear(rnn_size, 1)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.rnn_size, device=device),
                torch.zeros(batch_size, self.rnn_size, device=device))

    def forward(self, obs, goal, hx, cx, h_neighbors=None, comm_mask=None, agent_idx=0):

        x = F.relu(self.conv1(obs))
        x = self.res1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        flat = F.relu(x.view(x.size(0), -1))

        g = F.relu(self.goal_fc(goal))
        hidden_input = torch.cat([flat, g], dim=1)

        h1 = F.relu(self.fc1(hidden_input))
        h2 = self.fc2(h1)


        h3 = F.relu(h2 + hidden_input)


        if h_neighbors is not None and comm_mask is not None:


            info = self.attention(h_neighbors, attn_mask=comm_mask)

            my_info = info[:, agent_idx, :]


            combined = torch.cat([h3, my_info], dim=1)
            h3 = F.relu(self.comm_mixer(combined))


        hx, cx = self.lstm(h3, (hx, cx))

        policy_logits = self.policy_layer(hx)
        policy = F.softmax(policy_logits, dim=-1)
        policy_sig = torch.sigmoid(policy_logits)

        value = self.value_layer(hx)
        blocking = torch.sigmoid(self.blocking_layer(hx))
        on_goal = torch.sigmoid(self.on_goal_layer(hx))

        return NetworkOutput(policy, value, hx, cx, blocking, on_goal, policy_sig)
