# nn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def board_to_tensor(board):
    """
    convert board to (2, H, W) float tensor as before.
    """
    size = board.size
    t = torch.zeros((2, size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            c = board.tiles[i][j].colour
            if c == Colour.RED:
                t[0, i, j] = 1.0
            elif c == Colour.BLUE:
                t[1, i, j] = 1.0
    return t

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class HexNetPV(nn.Module):
    def __init__(self, board_size=11, n_blocks=3, channels=64):
        super().__init__()
        self.conv_in = ConvBlock(2, channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 2, H, W)
        x = self.conv_in(x)
        x = self.res_blocks(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)   # logits shape (B, H*W)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)   # in [-1,1]

        return p, v
