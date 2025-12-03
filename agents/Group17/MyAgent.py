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

def apply_move(board: Board, move: Move):
    """
        Applies a move to the board by setting tile colour and updating winner.
    """
    player = current_player(board)
    board.set_tile_colour(move.x, move.y, player)
    board.has_ended(player)  # updates internal winner state

def is_terminal(board : Board):
    """
        Checks wether a board state is terminal
    """
    return board.get_winner() is not None

def rollout_policy(board : Board):
    """
        TODO: Rollout policy to be implemented - simulate until terminal state, returning the winning colour
    """

    rollout_board = deepcopy(board)
    size = rollout_board.size
    center = size / 2

    def move_score(move : Move):
        # prefer moves towards center
        return -((move.x - center) ** 2 + (move.y - center) ** 2)
    
    while not is_terminal((rollout_board)):
        legal_moves = get_legal_moves(rollout_board)

        # give moves which are close to the center a better weight
        scored = [(move_score(m), m) for m in legal_moves]
        best_score = max(scored, key=lambda x: x[0])[0]
        best_moves = [m for (s, m) in scored if s == best_score]

        move = random.choice(best_moves)
        apply_move(rollout_board, move)

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
        self.player = player

        self.children = []
        self.untried_moves = get_legal_moves(self.board)

        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c : float = 1.41):
        # c is a constant that balances exploration and exploitation

        # first if there are any unvisited nodes choose them first
        for child in self.children:
            if child.visits == 0:
                return child
        
        # if not then use the upper confidence applied to trees - UCB1 but applied to trees (UCT)
        return max(
            self.children,
            key=lambda child: (
                (child.wins / child.visits)
                + c * math.sqrt((2 *math.log(self.visits)) / child.visits)
            )
        )
    
    def select_child(self):
        # TODO: UCT-RAVE implementation
        return self.best_child()

    def expand(self):
        move = self.untried_moves.pop()

        next_board = deepcopy(self.board)
        next_player = current_player(next_board)

        apply_move(next_board, move)
        
        child = MCTSNode(
            board=next_board,
            parent=self,
            move=move,
            player=next_player
        )
        self.children.append(child)
        return child

    def backpropogate(self, winner : Colour):
        self.visits += 1

        if winner is not None and self.player == winner:
            self.wins += 1
        if self.parent is not None:
            self.parent.backpropogate(winner)






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

        # MCTS
        """
        Jude - MCTS Psuedocode

        get the root node

        while the time elapsed < time limit:

            Phase 1 - Selection:
                while node isnt fully expanded and has children:
                    select node with the best UCT child

            Phase 2 - Expansion:
                if node isnt terminal:
                    pick a random untried move
                    apply said move to board
                    add child node of this new board state to the tree
                    set node = child node

            Phase 3 - Simultion / rollout:
                select winner by running rollout on this nodes board
            
            Phase 4 - Backpropagation:
                run backpropogate passing the winner and the node
            
            then return the move of the child with the most visits
                OR use one of the four child selection methods - in lecture slides

        """
        root = MCTSNode(deepcopy(board))

        TIME_LIMIT = 0.9
        end_time = time.time() + TIME_LIMIT

        while time.time() < end_time:

            # selection
            node = root
            while node.is_fully_expanded() and not is_terminal(node.board):
                # TODO: select child should be UCT-RAVE
                node = node.select_child()

            # expansion
            if not is_terminal(node.board):
                node = node.expand()
            
            # simulation / rollout

            # TODO (TEAM TASK - NN): Use neural_network_evaluate(node.board)
            # to guide rollout or replace rollout_policy entirely.
            nn_eval = neural_network_evaluate(node.board)

            # TODO: placeholder - someone implement rollout policy
            winner = rollout_policy(node.board)

            # backpropogation
            node.backpropogate(winner)
        
        # select move with the most visits
        best_move = max(
            root.children,
            key=lambda child: child.visits
        ).move

        return best_move

        

        

        
        

    