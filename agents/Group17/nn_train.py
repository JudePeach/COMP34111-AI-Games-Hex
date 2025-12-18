import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from copy import deepcopy
from collections import defaultdict

from nn import HexNetPV, board_to_tensor

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, pi, z):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, pi, z))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def get_legal_moves(board):
    legal = []
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour is None:
                legal.append((i, j))
    return legal

def make_move(board, move, player):
    new_board = deepcopy(board)
    x, y = move
    new_board.set_tile_colour(x, y, player)
    return new_board


def current_player(board, turn):
    # turn flips after each move
    return Colour.RED if turn % 2 == 0 else Colour.BLUE

class MCTSNode:
    def __init__(self, board, player):
        self.board = board
        self.player = player

        # statistics
        self.N = defaultdict(int)  # visit count
        self.W = defaultdict(float)  # total value
        self.Q = defaultdict(float)  # mean value
        self.P = {}  # prior probabilities

        self.children = {}  # move -> MCTSNode

    def is_leaf(self):
        return len(self.P) == 0  # not expanded yet

class MCTS:
    def __init__(self, model, board_size, sims=200, cpuct=1.4):
        self.model = model
        self.sims = sims
        self.cpuct = cpuct
        self.board_size = board_size

    def run(self, root_board, player):
        root = MCTSNode(root_board, player)

        for _ in range(self.sims):
            self.simulate(root)

        # generate visit distribution π
        visits = torch.zeros(self.board_size * self.board_size)

        legal = get_legal_moves(root_board)
        for (x, y) in legal:
            idx = x * self.board_size + y
            visits[idx] = root.N[(x, y)]

        if visits.sum() == 0:
            visits += 1

        pi = visits / visits.sum()
        return pi

    def simulate(self, node):
        board = node.board
        player = node.player

        # Check terminal state
        if board.has_ended(Colour.RED):
            return 1 if player == Colour.RED else -1
        if board.has_ended(Colour.BLUE):
            return 1 if player == Colour.BLUE else -1

        # Expand leaf node
        if node.is_leaf():
            state_tensor = board_to_tensor(board).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)

            policy = torch.softmax(policy_logits, dim=1).squeeze()

            legal = get_legal_moves(board)

            # Create P distribution only over legal moves
            node.P = {}
            for (x, y) in legal:
                idx = x * self.board_size + y
                node.P[(x, y)] = policy[idx].item()

            # Normalize
            total = sum(node.P.values())
            if total == 0:
                for m in node.P:
                    node.P[m] = 1 / len(node.P)
            else:
                for m in node.P:
                    node.P[m] /= total

            return value.item()

        # Select move with UCB
        best_score, best_move = -1e9, None
        for move in node.P:
            u = self.ucb_score(node, move)
            if u > best_score:
                best_score = u
                best_move = move

        move = best_move

        # Next board state
        new_board = make_move(board, move, player)
        next_player = Colour.RED if player == Colour.BLUE else Colour.BLUE

        if move not in node.children:
            node.children[move] = MCTSNode(new_board, next_player)

        v = self.simulate(node.children[move])

        # Backup
        node.N[move] += 1
        node.W[move] += v
        node.Q[move] = node.W[move] / node.N[move]

        return v

    def ucb_score(self, node, move):
        c = self.cpuct
        N_sum = sum(node.N[m] for m in node.P)
        q = node.Q[move]
        p = node.P[move]
        u = c * p * math.sqrt(N_sum + 1) / (1 + node.N[move])
        return q + u

def self_play_game(model, board_size, mcts_sims=200):
    model.eval()
    buffer_entries = []

    board = Board(board_size)
    player = Colour.RED
    turn = 0

    mcts = MCTS(model, board_size, sims=mcts_sims)

    while True:
        pi = mcts.run(board, player)

        state = board_to_tensor(board)
        buffer_entries.append((state, pi, player))

        # Sample move from π
        move_index = torch.multinomial(pi, 1).item()
        x, y = divmod(move_index, board_size)

        board.set_tile_colour(x, y, player)

        # Check win
        if board.has_ended(player):
            winner = player
            break

        # Next player
        player = Colour.RED if player == Colour.BLUE else Colour.BLUE
        turn += 1

    training_data = []
    for state, pi, p in buffer_entries:
        z = 1 if p == winner else -1
        training_data.append((state, pi, z))

    return training_data


# ============================================================
# TRAINING LOOP
# ============================================================

def train_hex_network(
    board_size=11,
    iterations=50,
    games_per_iter=5,
    batch_size=64,
    mcts_sims=200
):
    model = HexNetPV(board_size=board_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    replay = ReplayBuffer(capacity=50000)

    for it in range(iterations):
        print(f"\n=== Iteration {it+1}/{iterations} ===")

        # 1. Self-play
        for g in range(games_per_iter):
            game_data = self_play_game(model, board_size, mcts_sims)
            for entry in game_data:
                replay.add(*entry)
            print(f"  Game {g+1}/{games_per_iter} generated {len(game_data)} samples.")

       # 2. Training
        model.train()
        if len(replay) < batch_size:
            continue

        batch = replay.sample(batch_size)
        states = torch.stack([s for (s, _, _) in batch])
        pis = torch.stack([p for (_, p, _) in batch])
        zs = torch.tensor([z for (_, _, z) in batch], dtype=torch.float32)

        policy_logits, values = model(states)

        # Policy loss
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(pis * log_probs, dim=1))

        # Value loss
        value_loss = nn.functional.mse_loss(values, zs)

        # Total loss
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}  (Policy {policy_loss.item():.4f}, Value {value_loss.item():.4f})")

        # Save checkpoint
        if (it + 1) % 5 == 0:
            torch.save(model.state_dict(), "hex_model.pt")
            print("  Saved checkpoint to hex_model.pt")

    torch.save(model.state_dict(), "hex_model_final.pt")
    print("Training complete! Final model saved to hex_model_final.pt")

    return model


if __name__ == "__main__":
    train_hex_network()
