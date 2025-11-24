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

# util funcs
def current_player(board : Board):

    """
        Returns who the current player is given the current board state.
    """

    red, blue = 0, 0
    for row in board.tiles:
        for tile in row:
            if tile.colour == RED:
                red += 1
            else:
                blue += 1
    return Colour.RED if red == blue else return Colour.BLUE

def get_legal_moves(board : Board):

    """
        Gets the availavle legal moves for the current player given the current board state
    """

    moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board[i][j].colour == Colour.EMPTY:
                moves.append(Move(i, j))
    return moves

def apply_move(board: Board, move: Move):
    """
        Applies a move to the board by setting tile colour and updating winner.
    """
    player = current_player(board)
    board.set_tile_colour(move.i, move.j, player)
    board.has_ended(player)  # updates internal winner state

def is_terminal(boar : Board):
    """
        Checks wether a board state is terminal
    """
    return board.get_winner() is not None

class MCTSNode():
    """
        Monte Carlo Tree Search Node
    """
    def __init__(self,
        board : Board,
        parent : Optional[MCTSNode] = None,
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

    def is_fully_expanded(self);
        return len(self.untried_moves) == 0

    def best_child(self, c : float = 1.41):
        return max(
            self.children,
            key=lambda child: (
                child.wins / child.visits
                + c * math.sqrt(math.log(self.visits) / child.visits)
            )
        )

    def expand(self):
        move = self.untried_moves.pop()
        next_board = deepcopy(self.board)
        next_player = current_player(next_board)

        apply_move(next_
        
        child = MCTSNode(
            board=next_board,
            parent=self,
            move=move,
            player=next_player
        )
        self.children.append(child)
        return childboard, move)

    def backpropogate(self, winner : Colour):
        self.visits += 1
        if self.player == winner:
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

        # UCT + RAVE

        # Rollout 

        # NN

        # Placeholder 
        return Move(-1, -1)

    