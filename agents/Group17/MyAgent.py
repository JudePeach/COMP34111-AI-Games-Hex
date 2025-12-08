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

# util funcs
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
        TODO: Neural Network evaluation function to be implemented - think should return the probability of winning for the current player

    """
    # placeholder
    return 0.5

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

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c : float = 1.41):
        # c is a constant that balances exploration and exploitation

        # first if there are any unvisited children choose randomly among them
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)

        # all children visited -> use UCT
        def uct_score(child):
            # exploitation
            exploitation = child.wins / child.visits
            # exploration
            exploration = c * math.sqrt((2 * math.log(self.visits)) / child.visits)
            return exploitation + exploration

        return max(self.children, key=uct_score)
    
    def select_child(self):
        # TODO: UCT-RAVE implementation
        return self.best_child()

    def expand(self):
        move = self.untried_moves.pop()

        next_board = deepcopy(self.board)

        move_player = self.next_player

        apply_move_fast(next_board, move, move_player)
        
        child = MCTSNode(
            board=next_board,
            parent=self,
            move=move,
            player=move_player
        )
        self.children.append(child)
        return child

    def backpropogate(self, winner : Colour):
        node = self

        while node is not None:
            node.visits += 1
            if winner == node.player:
                node.wins += 1
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
        root = MCTSNode(deepcopy(board))
        cp = current_player(board)
        root.player = Colour.RED if cp == Colour.BLUE else Colour.BLUE

        TIME_LIMIT = 0.9
        end_time = time.time() + TIME_LIMIT

        while time.time() < end_time:

            # selection
            node = root
            while node.is_fully_expanded() and not is_terminal(node.board):
                # TODO: select child should be UCT-RAVE
                node = node.select_child()

            # expansion
            if not is_terminal(node.board):
                if not node.is_fully_expanded():
                    node = node.expand()
            
            # simulation / rollout
            winner = heuristic_rollout_policy(node.board)

            # backpropogation
            node.backpropogate(winner)

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
