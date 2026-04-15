

class Board(object):
    """ Board for the game Caro """

    def __init__(self, rows: int = 15, cols: int = 15, winning_condition: int = 5, XO: int = 1):

        """ 
        Parameters
        ----------
        rows: int
            Number of rows of the board (default is 15)
        cols: int
            Number of columns of the board (default is 15)
        winning_condition: int
            Number of pieces in a row needed to win the game (default is 5)
        XO: int
            The player who will start the game, 1 for X and -1 for O (default is 1/X)
        """

        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

        self.winning_condition = winning_condition

        self.originXO = XO
        self.turn = XO
        self.your_turn = XO
        self.AI_turn = -XO
        """ default: the human player starts first """

        self.hardness = 0
        self.is_use_AI = False
        """ default: the game is played between two human players """

        self.move_history = []  # List to store the history of move

    def reset(self):
        """ Reset the board to the initial state """
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.turn = self.originXO

        self.your_turn = self.originXO
        self.AI_turn = -self.originXO
        
        self.move_history = []  # Clear the move history

    def get_possible_moves(self):
        """ Get a list of all possible moves (empty cells) on the board """
        possible_move = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 0:
                    possible_move.append((i, j))
    
        return possible_move
    
    def is_draw(self):

        """ Check if the game is a draw (no more possible moves) """

        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 0:
                    return False
                
        return True
    
    def make_move(self, x : int, y : int):
        """ Make a move on the board at the specified coordinates (x, y) for the current player """
        
        if self.grid[x][y] != 0:
            return False
        
        self.grid[x][y] = self.turn
        self.move_history.append((x, y, self.turn))  # Add the move to the history
        self.turn = -self.turn  # Switch turns  

        return True
    
    def undo_move(self):
        """ Undo the last move made on the board """
        if not self.move_history:
            return False  # No moves to undo
        
        last_move = self.move_history.pop()  # Get the last move
        x, y, player = last_move
        self.grid[x][y] = 0  # Clear the cell
        self.turn = player  # Switch back to the player who made the move

        return True
    
    def getRow(self, x: int):
        """ Get entire row at index x """
        return self.grid[x]
    
    def getCol(self, y: int):
        """ Get entire column at index y """
        return [self.grid[i][y] for i in range(self.rows)]
    
    def getMainDiagonal(self, x: int, y: int):
        diag = []
        for i in range(-self.winning_condition + 1, self.winning_condition):
            new_x = x + i
            new_y = y + i
            if 0 <= new_x < self.rows and 0 <= new_y < self.cols:
                diag.append(self.grid[new_x][new_y])
        
        return diag
    
    def getAntiDiagonal(self, x: int, y: int):
        anti_diag = []
        for i in range(-self.winning_condition + 1, self.winning_condition):
            new_x = x + i
            new_y = y - i
            if 0 <= new_x < self.rows and 0 <= new_y < self.cols:
                anti_diag.append(self.grid[new_x][new_y])
        
        return anti_diag
    
    def count_consecutive(self, line: list, player: int):
        max_count = 0
        current_count = 0

        for cell in line:
            if cell == player:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def check_winning(self, x: int, y: int, player: int) -> bool:
        """
        Check if placing a piece at (x, y) for the given player results in a win
        Returns True if it's a winning move, False otherwise
        """
        # Get all 4 directions
        row = self.grid[x]
        col = [self.grid[i][y] for i in range(self.rows)]
        main_diag = self.getMainDiagonal(x, y)
        anti_diag = self.getAntiDiagonal(x, y)
        
        # Check each direction for winning condition
        if self.count_consecutive(row, player) >= self.winning_condition:
            return True
        if self.count_consecutive(col, player) >= self.winning_condition:
            return True
        if self.count_consecutive(main_diag, player) >= self.winning_condition:
            return True
        if self.count_consecutive(anti_diag, player) >= self.winning_condition:
            return True
        
        return False

    def get_winner(self):
        """
        Return the winner of the current board state
            1 if the winner is X
            -1 if the winner is O
            0 if there is no winner yet
            2 if the game is a draw
        """

        if not self.move_history:
            return 0  # No moves made yet, so no winner
        
        last_move = self.move_history[-1]  # Get the last move
        x, y, player = last_move

        if self.check_winning(x, y, player):
            return player
        
        if self.is_draw():
            return 2  # Game is a draw
        
        return 0  # No winner yet
    
    def set_AI_move_first(self):
        """ Set the AI to move first in the game """
        self.your_turn = -self.originXO
        self.AI_turn = self.originXO

    def setHardAI(self, hardness: int):
        """ Set the hardness level of the AI (0 for easy, 1 for medium, 2 for hard) """
        self.hardness = hardness
        self.is_use_AI = True

    def set_use_AI(self, use_AI: bool):
        """ Set whether to use AI in the game or not """
        self.is_use_AI = use_AI

    def set_Human_vs_Human(self):
        """ Set the game mode to human vs human """
        self.is_use_AI = False

    def get_current_XO_for_AI(self):
        return self.AI_turn