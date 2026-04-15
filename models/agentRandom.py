from game.board import Board
import copy 
import random
import time

class AgentRandom:
    """ Agent that selects a random move from the available moves on the board """

    def __init__(self, board: Board):
        self.board = board

    def get_move(self):

        """ Get a random move from the available moves on the board """
        possible_moves = self.board.get_possible_moves()
        if not possible_moves:
            return None  # No moves available
    
        time.sleep(1)
        return random.choice(possible_moves)

