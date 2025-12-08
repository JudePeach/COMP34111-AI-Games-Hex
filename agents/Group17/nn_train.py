import torch
import torch.nn as nn
import torch.optim as optim
import random

from nn import HexNet, board_to_tensor
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# Create synthetic training data (replace with self-play later)
def generate_random_board(board_size=11):
    """
    Makes a random board for training.
    You will replace this with actual self-play game states later.
    """
    board = Board(board_size)
    num_moves = random.randint(5, board_size * board_size - 1)

    turn = Colour.RED
    for _ in range(num_moves):
        empties = [(i, j) for i in range(board_size) for j in range(board_size)
                   if board.tiles[i][j].colour is None]
        if not empties:
            break

        x, y = random.choice(empties)
        board.set_tile_colour(x, y, turn)
        turn = Colour.BLUE if turn == Colour.RED else Colour.RED

    # Assign a random "winner" target for now - replace with real results later
    target = torch.tensor([[random.random()]], dtype=torch.float32)

    return board, target

def train_hex_network(epochs=5, batches_per_epoch=200, board_size=11):
    model = HexNet(board_size=board_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):

        for _ in range(batches_per_epoch):
            # Generate synthetic training sample
            board, target = generate_random_board(board_size)
            inp = board_to_tensor(board).unsqueeze(0)  # add batch dim

            pred = model(inp)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss = {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "hex_model.pt")
    print("✔ Saved model to hex_model.pt")

    return model

if __name__ == "__main__":
    train_hex_network()
