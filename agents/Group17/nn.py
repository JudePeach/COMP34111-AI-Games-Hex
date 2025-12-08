import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

def board_to_tensor(board):
    """ 
        Converts the current board state to a tensor rep
    """

    size = board.size
    tensor = torch.zeros((2, size, size))

    for i in range(size):
        for j in range(size):
            tile = board.tiles[i][j].colour 
            if tile == Colour.RED:
                tensor[0][i][j] = 1
            elif tile == Colour.BLUE:
                tensor[1][i][j] = 1
    
    return tensor

class HexNet(nn.Module):
    def __init__(self, board_size=11):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(64 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)   # output probability
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # probability in [0,1]
        return x

def train_dummy_model():
    model = HexNet(board_size=11)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Dummy data: 1000 random boards with random win prob targets
    for epoch in range(5):
        for _ in range(100):
            # random board tensor
            board_tensor = torch.randn((1, 2, 11, 11))
            
            # random probability label
            target = torch.rand((1, 1))

            # forward
            pred = model(board_tensor)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch", epoch, "Loss:", loss.item())

    return model