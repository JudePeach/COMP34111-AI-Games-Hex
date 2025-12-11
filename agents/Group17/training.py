# hex_alphazero_trainer.py
"""
Single-file AlphaZero-like trainer for Hex.
Usage (quick test):
    python hex_alphazero_trainer.py --board_size 7 --iterations 2 --games_per_iter 2 --mcts_sims 50

Defaults are small to let you smoke-test; increase sims/games/iterations for real training.
"""
import argparse
import math
import random
import os
import sys
from copy import deepcopy
from collections import deque, defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ensure src is importable (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# -----------------------------
# Model
# -----------------------------
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
    def __init__(self, board_size=11, n_blocks=4, channels=64):
        super().__init__()
        self.board_size = board_size
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
        # x: (B, 2, H, W) where channel 0 = current player's stones, channel1 = opponent
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

# -----------------------------
# Helpers: board -> tensor (canonicalized)
# -----------------------------
def board_to_tensor(board: Board, player_to_move: Colour):
    """
    Returns (2, H, W) float32 tensor where channel 0 = current player's stones, channel 1 = opponent's stones.
    This canonicalization ensures model always sees current player on channel 0.
    """
    size = board.size
    t = torch.zeros((2, size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            c = board.tiles[i][j].colour
            if c is None:
                continue
            if c == player_to_move:
                t[0, i, j] = 1.0
            else:
                t[1, i, j] = 1.0
    return t

# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, pi, z):
        # state: tensor (2,H,W), pi: 1D numpy array over H*W, z: float in [-1,1]
        self.buffer.append((state.clone(), pi.copy(), float(z)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([item[0] for item in batch])  # (B,2,H,W)
        pis = torch.tensor([item[1] for item in batch], dtype=torch.float32)  # (B, H*W)
        zs = torch.tensor([item[2] for item in batch], dtype=torch.float32)  # (B,)
        return states, pis, zs

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# MCTS Node & MCTS
# -----------------------------
class MCTSNode:
    def __init__(self, board: Board, player: Colour):
        self.board = board
        self.player = player  # player to move at this node
        self.P = {}  # prior prob for legal moves (move tuple -> prob)
        self.N = defaultdict(int)  # visit counts per move
        self.W = defaultdict(float)  # total value per move
        self.Q = defaultdict(float)  # mean value per move
        self.children = {}  # move -> child MCTSNode

    def is_expanded(self):
        return len(self.P) > 0

class MCTS:
    def __init__(self, model: nn.Module, board_size:int, device, sims=400, cpuct=1.4):
        self.model = model
        self.board_size = board_size
        self.sims = sims
        self.cpuct = cpuct
        self.device = device

    def run(self, root_board: Board, root_player: Colour, add_root_noise=True):
        """
        Returns visit distribution pi (numpy array length board_size*board_size).
        """
        root = MCTSNode(deepcopy(root_board), root_player)

        # Expand root to obtain initial P and value
        self._expand_and_eval(root)

        # Add Dirichlet noise to root priors for exploration (AlphaZero)
        if add_root_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.sims):
            self._simulate(root)

        # Build visit count distribution
        visits = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        legal = get_legal_moves(root.board)
        for (x, y) in legal:
            visits[x * self.board_size + y] = root.N[(x, y)]
        if visits.sum() == 0:
            visits += 1.0
        pi = visits / visits.sum()
        return pi, root

    def _simulate(self, root_node: MCTSNode):
        node = root_node
        path = []  # sequence of (node, move)
        # selection
        while node.is_expanded():
            # pick best move by UCB
            best_score = -1e9
            best_move = None
            for move in node.P.keys():
                score = self._ucb(node, move)
                if score > best_score:
                    best_score = score
                    best_move = move
            if best_move is None:
                # no legal moves (shouldn't happen)
                return 0.0
            path.append((node, best_move))
            # step to child
            if best_move not in node.children:
                # create child node lazily
                new_board = deepcopy(node.board)
                make_move_on_board(new_board, best_move, node.player)
                next_player = Colour.RED if node.player == Colour.BLUE else Colour.BLUE
                node.children[best_move] = MCTSNode(new_board, next_player)
            node = node.children[best_move]

            # terminal check
            if node.board.get_winner() is not None:
                # game finished; determine return from node.player's view
                winner = node.board.get_winner()
                if winner == node.player:
                    leaf_value = 1.0
                else:
                    leaf_value = -1.0
                # backup
                self._backup(path, leaf_value)
                return leaf_value

        # now node is a leaf (not expanded)
        value = self._expand_and_eval(node)  # value is in [-1,1] for node.player
        # backup value up the path (value is from node.player's POV)
        self._backup(path, value)
        return value

    def _expand_and_eval(self, node: MCTSNode):
        """
        Expands node by setting node.P over legal moves and returns value for node.player in [-1,1].
        """
        board = node.board
        player = node.player

        # terminal?
        winner = board.get_winner()
        if winner is not None:
            return 1.0 if winner == player else -1.0

        # Evaluate with neural net
        state_tensor = board_to_tensor(board, player).unsqueeze(0).to(self.device)  # (1,2,H,W)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()  # (H*W,)
            value = value.item()  # in [-1,1]

        # Build prior over legal moves only
        legal = get_legal_moves(board)
        priors = {}
        # apply softmax over logits but mask illegal moves
        # compute softmax in a numerically stable way
        logits = policy_logits.copy()
        # mask illegal with -inf by setting large negative
        mask = np.full(self.board_size * self.board_size, -1e9, dtype=np.float32)
        for (x, y) in legal:
            mask[x * self.board_size + y] = 0.0
        masked = logits + mask
        exp = np.exp(masked - np.max(masked))
        probs = exp / (exp.sum() + 1e-16)
        for (x, y) in legal:
            priors[(x, y)] = float(probs[x * self.board_size + y])
        # fallback if probs sum to 0
        if sum(priors.values()) <= 0:
            for m in priors:
                priors[m] = 1.0 / len(priors)

        node.P = priors
        return value  # perspective: node.player

    def _ucb(self, node: MCTSNode, move):
        """
        UCB + PUCT
        """
        # exploitation
        q = node.Q[move]
        # exploration
        N_sum = sum(node.N[m] for m in node.P)
        p = node.P[move]
        u = self.cpuct * p * math.sqrt(N_sum + 1.0) / (1 + node.N[move])
        return q + u

    def _backup(self, path, leaf_value):
        """
        path: list of (node, move) where node is the parent node and move was chosen at that parent
        leaf_value is value from the FINAL node.player perspective. We must flip sign as we go up.
        """
        v = leaf_value
        # iterate from last (closest to leaf) back to root
        for (node, move) in reversed(path):
            # node.player is the player who moved to create the child node at this step?
            # Convention: node.player is the player to move at this node. The move was played by that player.
            # We store statistics from the viewpoint of node.player (the one who took the move).
            node.N[move] += 1
            node.W[move] += v
            node.Q[move] = node.W[move] / node.N[move]
            # flip perspective for parent
            v = -v

    def _add_dirichlet_noise(self, root_node: MCTSNode, alpha=0.3, eps=0.25):
        """
        Add Dirichlet noise to priors at the root to encourage exploration.
        alpha and eps can be tuned by board size: larger board -> maybe smaller alpha.
        """
        legal = list(root_node.P.keys())
        if not legal:
            return
        noise = np.random.dirichlet([alpha] * len(legal))
        for i, m in enumerate(legal):
            root_node.P[m] = (1 - eps) * root_node.P.get(m, 0.0) + eps * noise[i]

# -----------------------------
# Utility functions & wrappers
# -----------------------------
def get_legal_moves(board: Board):
    legal = []
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour is None:
                legal.append((i, j))
    return legal

def make_move_on_board(board: Board, move, player: Colour):
    x, y = move
    board.set_tile_colour(x, y, player)

def augment_state_pi(state_tensor: torch.Tensor, pi: np.ndarray, board_size: int):
    """
    Return list of augmented (state, pi) pairs using simple symmetries: horizontal flip and transpose.
    pi is numpy array length board_size*board_size in row-major (x*W + y) ordering.
    """
    # reshaped pi to (H, W)
    pi_grid = pi.reshape((board_size, board_size))
    aug = []
    # identity
    aug.append((state_tensor.clone(), pi_grid.copy()))
    # horizontal flip (mirror left-right)
    s = torch.flip(state_tensor, dims=[2])  # flip columns (W) dimension index is 2 or 3? state (2,H,W): dims 2 is H, 3 is W -> but here shape (2,H,W) so dims [2] is W
    p = np.fliplr(pi_grid)
    aug.append((s.clone(), p.copy()))
    # vertical flip
    s2 = torch.flip(state_tensor, dims=[1])  # flip H
    p2 = np.flipud(pi_grid)
    aug.append((s2.clone(), p2.copy()))
    # transpose (swap axes)
    s3 = state_tensor.permute(0,2,1).clone()  # (2,W,H) swapped
    p3 = pi_grid.T.copy()
    aug.append((s3, p3))
    # Note: some of these transforms may not be exact Hex symmetries depending on orientation; these are simple augmentations.
    # Convert p back to flattened
    out = []
    for s_t, p_g in aug:
        out.append((s_t, p_g.reshape(-1)))
    return out

# -----------------------------
# Self-play
# -----------------------------
def self_play_game(model, board_size, mcts_sims, device, temp_gamma=1.0):
    """
    Plays a single self-play game using MCTS guided by model and returns list of (state_tensor, pi, z).
    state_tensor is canonicalized to the player to move.
    z is in {-1, +1} from the perspective of the player who was to move at that state.
    """
    model.eval()
    mcts = MCTS(model, board_size, device, sims=mcts_sims)
    board = Board(board_size)
    player = Colour.RED  # Red moves first (as in your code base)
    training_examples = []  # (state_tensor, pi, player)

    turn = 0
    while True:
        # run MCTS from current root
        pi, root = mcts.run(board, player, add_root_noise=True)

        # store state and pi (canonicalized: state_tensor uses player as channel 0)
        state_tensor = board_to_tensor(board, player)  # (2,H,W)
        training_examples.append((state_tensor, pi.copy(), player))

        # sample move from pi with temperature schedule
        # temperature: use high randomness early, deterministic later
        T = 1.0 if turn < 10 else 0.1  # simple schedule: first 10 moves more exploratory
        if T == 0:
            # choose argmax
            move_index = int(np.argmax(pi))
        else:
            # apply temperature: pi^(1/T)
            logits = np.log(np.maximum(pi, 1e-12)) / T
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            move_index = np.random.choice(len(probs), p=probs)

        x, y = divmod(move_index, board_size)
        make_move_on_board(board, (x, y), player)

        # check win
        winner = board.get_winner()
        if winner is not None:
            # assign z for each stored state: +1 if the player to move at that state ended up winning, else -1
            examples = []
            for (s, p, p_player) in training_examples:
                z = 1.0 if p_player == winner else -1.0
                examples.append((s, p, z))
            return examples

        # next player
        player = Colour.RED if player == Colour.BLUE else Colour.BLUE
        turn += 1
        # safety cap (shouldn't happen)
        if turn > board_size * board_size:
            # draw-ish fallback (rare in Hex)
            examples = []
            for (s, p, p_player) in training_examples:
                examples.append((s, p, 0.0))
            return examples

# -----------------------------
# Training loop
# -----------------------------
def train(
    board_size=11,
    iterations=50,
    games_per_iter=10,
    mcts_sims=200,
    batch_size=64,
    replay_capacity=50000,
    lr=1e-3,
    device=None,
    checkpoint_dir="checkpoints"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = HexNetPV(board_size=board_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity)

    for it in range(1, iterations + 1):
        print(f"\n=== Iteration {it}/{iterations} ===")
        # Self-play
        generated = 0
        for g in range(games_per_iter):
            examples = self_play_game(model, board_size, mcts_sims, device)
            # store with augmentation
            for (state, pi, z) in examples:
                # simple augmentation
                for s_aug, p_aug in augment_state_pi(state, pi, board_size):
                    replay.add(s_aug.to(device), p_aug.copy(), z)
            generated += len(examples)
            print(f"  Game {g+1}/{games_per_iter}: produced {len(examples)} examples.")
        print(f"  Generated {generated} examples; replay size now {len(replay)}")

        # Training
        if len(replay) < batch_size:
            print("  Not enough samples to train; continue.")
            continue

        model.train()
        # number of training steps per iteration - tune as needed
        train_steps = max(1, len(replay) // batch_size // 4)
        for step in range(train_steps):
            states, pis, zs = replay.sample(batch_size)
            states = states.to(device)
            pis = pis.to(device)
            zs = zs.to(device)

            policy_logits, values = model(states)  # logits shape (B, H*W), values shape (B,)
            # policy loss: cross-entropy with target pi
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(pis * log_probs, dim=1))
            # value loss (MSE)
            # target zs in {-1,1}; values in [-1,1]
            value_loss = F.mse_loss(values, zs)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        print(f"  Training done (steps={train_steps}); loss policy={policy_loss.item():.4f}, value={value_loss.item():.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"hex_model_iter{it}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint {ckpt_path}")

    # final save
    final_path = os.path.join(checkpoint_dir, "hex_model_final_new.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    return model

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--board_size", type=int, default=11)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--games_per_iter", type=int, default=10)
    p.add_argument("--mcts_sims", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--replay_capacity", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # quick randomness seeds for deterministic debugging (optional)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(
        board_size=args.board_size,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
