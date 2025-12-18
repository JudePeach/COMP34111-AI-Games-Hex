from random import choice
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from copy import deepcopy
# import for type hinting
from typing import Optional, List

import math
import random
import time

import torch
import torch.nn as nn

from agents.Group17.training import HexNetPV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# util funcs

def evaluate_node(node):
    # value returned is probability RED wins
    red_win_prob = neural_network_evaluate(node.board)
    
    # convert to probability that *node.next_player* wins
    if node.next_player == Colour.RED:
        return red_win_prob
    else:
        return 1 - red_win_prob


def load_hex_model(path = "/home/hex/agents/Group17/checkpoints/hex_model_final_new.pt"):
    """
        Loads our trained neural network model from file.
    """
    state_dict = torch.load(path, map_location=DEVICE)

    model = HexNetPV()  
    model.load_state_dict(state_dict)
    model.eval()
    return model

hex_model = load_hex_model()

def board_to_tensor(board : Board):
    size = board.size
    tensor = torch.zeros((2, size, size), dtype=torch.float32)

    for i in range(size):
        for j in range(size):
            tile = board.tiles[i][j].colour
            if tile == Colour.RED:
                tensor[0, i, j] = 1   # channel 0 = RED
            elif tile == Colour.BLUE:
                tensor[1, i, j] = 1   # channel 1 = BLUE
    return tensor.unsqueeze(0)  # shape: (1, 2, size, size)

def current_player(board : Board):

    """
        Returns who the current player is given the current board state.
    """

    red, blue = 0, 0
    for row in board.tiles:
        for tile in row:
            if tile.colour == Colour.RED:
                red += 1
            elif tile.colour == Colour.BLUE:
                blue += 1
    return Colour.RED if red == blue else Colour.BLUE

def get_legal_moves(board : Board):

    """
        Gets the availavle legal moves for the current player given the current board state
    """

    moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour is None:
                moves.append(Move(i, j))
    return moves

def apply_move_fast(board: Board, move: Move, player : Colour):
    """
        Applies a move to the board by setting tile colour and updating winner.
    """
    board.set_tile_colour(move.x, move.y, player)
    try:
        board.has_ended(player)
    except Exception:
        # ignore if has_ended behaves differently
        pass

def apply_move(board : Board, move : Move):
    """
        Backwards-compatible apply_move using current_player (slower).
    """
    player = current_player(board)
    board.set_tile_colour(move.x, move.y, player)
    try:
        board.has_ended(player)
    except Exception:
        pass
    

def is_terminal(board : Board):
    """
        Checks wether a board state is terminal
    """
    return board.get_winner() is not None

def rollout_policy(board : Board):
    """ 
        Shuffle the legal moves and play them in sequence (fast rollout).
        Uses deepcopy(board) and passes player correctly to apply_move_fast.
    """

    # make a fast local copy
    rollout_board = deepcopy(board)

    # determine starting player once
    player = current_player(rollout_board)

    legal_moves = get_legal_moves(rollout_board)
    random.shuffle(legal_moves)

    for move in legal_moves:
        # always pass the player who is about to move
        apply_move_fast(rollout_board, move, player)

        # alternate the player
        player = Colour.RED if player == Colour.BLUE else Colour.BLUE

        if is_terminal(rollout_board):
            break
    
    return rollout_board.get_winner()

def heuristic_rollout_policy(board: Board):
    rollout_board = deepcopy(board)
    player = current_player(rollout_board)

    for _ in range(board.size * board.size):
        legal = get_legal_moves(rollout_board)
        if not legal:
            break

        # score moves
        scored = []
        for m in legal:
            # distance to opposite edge (encourage progress)
            if player == Colour.RED:
                score = board.size - m.x
            else:
                score = board.size - m.y

            # tiny randomness to avoid determinism
            score += random.random() * 0.1

            scored.append((score, m))

        # pick highest-scoring move
        _, best = max(scored, key=lambda x: x[0])
        apply_move_fast(rollout_board, best, player)

        player = Colour.RED if player == Colour.BLUE else Colour.BLUE

        if is_terminal(rollout_board):
            break

    return rollout_board.get_winner()


def neural_network_evaluate(board : Board):
    """
        Neural Network evaluation function to be implemented - returns the probability of winning for the current player

    """

    tensor = board_to_tensor(board).to(DEVICE)
    with torch.no_grad():
        policy_logits, value = hex_model(tensor)  # depends on your model output
        value = value.item()
    return value  # probability that current player wins

def key_board(board : Board):
    """
        Creates a hashable representation of a board state - for use in RAVE transposition table
    """
    return tuple(tuple(tile.colour for tile in row) for row in board.tiles)


class MCTSNode():
    """
        Monte Carlo Tree Search Node
    """
    def __init__(self,
        board : Board,
        parent : Optional["MCTSNode"] = None,
        move : Optional[Move] = None,
        player : Optional[Colour] = None
        ):

        self.board = board
        self.parent = parent
        self.move = move
        
        # 'player' is the player who MADE the move that produced this node
        self.player = player
        # next_player is the player to move in this node
        self.next_player = current_player(self.board)

        self.children = []
        self.untried_moves = get_legal_moves(self.board)

        self.visits = 0
        self.wins = 0.0

        # RAVE stats
        self.rave_stats = {}

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c : float = 1.41, b : float = 0.1):
        """
            UCT with RAVE best child selection
        """
        # c is a constant that balances exploration and exploitation

        # first if there are any unvisited children choose randomly among them
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)

        # all children visited -> use UCT
        def uct_rave_score(child):
            """
                Calculates the UCT-RAVE score for a child node. RAVE adds an additional est for good a move is based not only on the childs stats, but also:
                    - results of all playouts playing said move - allows learning which moves are better, much earlier in the search -> Speed up

            """
            # exploitation
            exploitation = child.wins / child.visits
            # exploration
            exploration = c * math.sqrt((2 * math.log(self.visits)) / child.visits)

            # rave value
            if child.move in self.rave_stats:
                rave_wins, rave_visits = self.rave_stats[child.move]
                rave_value = rave_wins / rave_visits if rave_visits > 0 else 0 # accounting for divide by zero error
            else:
                rave_value = 0
                rave_visits = 0
            
            # beta weighting
            beta = rave_visits / (child.visits + rave_visits + 4 * child.visits * rave_visits * (b ** 2))
            blended = (1 - beta) * exploitation + beta * rave_value
            return blended + exploration

        return max(self.children, key=uct_rave_score)
    
    def select_child(self):
        # UCT-RAVE implementation in best child function
        return self.best_child()

    def expand(self, transposition_table = None):
        move = self.untried_moves.pop()

        next_board = deepcopy(self.board)

        move_player = self.next_player

        apply_move_fast(next_board, move, move_player)

        key = key_board(next_board)

        # if the boards in the transp table can reuse the node
        if transposition_table is not None and key in transposition_table:
            child = transposition_table[key]
            child.parent = self
        else:
            child = MCTSNode(
                board=next_board,
                parent=self,
                move=move,
                player=move_player
            )
            if transposition_table is not None: # add node to tt table 
                transposition_table[key] = child
        self.children.append(child)
        return child

    def backpropogate(self, winner : Colour, value : float):
        """
            Backprop, with AMAF/RAVE updates also
        """
        node = self

        played_moves = []
        cur = self
        while cur is not None:
            played_moves.append(cur.move)
            cur = cur.parent

        while node is not None:
            node.visits += 1
            if winner == node.player:
                #node.wins += 1
                node.wins += value if node.player == Colour.RED else (1 - value)

            
            # rave updating
            for mov in played_moves:
                if mov not in node.rave_stats:
                    node.rave_stats[mov] = [0, 0] # (0 wins 0 visits)
                if winner == node.player:
                    node.rave_stats[mov][0] += 1
                node.rave_stats[mov][1] += 1

            node = node.parent






class MyAgent(AgentBase):

    """
        Group 17 agent implementation
    """

    _choices : list[Move]
    _board_size : int = 11

    def __init__(self, colour: Colour):
        """
            Constructor 
        """
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
    
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
            Group 17 move making method.
        """

        transposition_table = {}

        root = MCTSNode(deepcopy(board))
        cp = current_player(board)
        root.player = Colour.RED if cp == Colour.BLUE else Colour.BLUE

        TIME_LIMIT = 0.9
        end_time = time.time() + TIME_LIMIT

        while time.time() < end_time:

            # selection
            node = root
            while node.is_fully_expanded() and not is_terminal(node.board):
                node = node.select_child()

            # expansion
            if not is_terminal(node.board):
                if not node.is_fully_expanded():
                    node = node.expand(transposition_table)
            
            # simulation / rollout
            #winner = heuristic_rollout_policy(node.board)
            value = evaluate_node(node)
            winner = node.next_player if value > 0.5 else (
                Colour.RED if node.next_player == Colour.BLUE else Colour.BLUE
            )


            # backpropogation
            node.backpropogate(winner, value)

        if not root.children:
            legal = get_legal_moves(board)
            if not legal:
                return Move(0, 0)
            return random.choice(legal)
        
        # select move with the most visits
        best_move = max(
            root.children,
            key=lambda child: child.visits
        ).move

        return best_move
