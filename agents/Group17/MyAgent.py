from random import choice 

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


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

    