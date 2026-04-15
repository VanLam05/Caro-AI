from game.board import Board
import math

""" Agent that uses the Minimax algorithm with alpha-beta pruning to select the best move based on the current board state """

class AgentMiniMax:
    def __init__(self, board: Board, max_depth: int = 3):
        self.board = board
        self.max_depth = max_depth
        self.killer_moves = [set() for _ in range(max_depth + 2)]  # Killer move heuristic
        self.transposition_table = {}  # Transposition table for memoization
        self.evaluation_count = 0  # Statistics

    def get_candidate_moves(self):
        """Get a list of candidate moves with larger radius for better play"""
        candidate_move = set()
        
        if not self.board.move_history:
            # First move - play in center
            center = self.board.rows // 2
            return [(center, center)]
        
        # Look at last 20 moves with larger radius for more strategic options
        for move in self.board.move_history[-20:]:
            if isinstance(move, (tuple, list)):
                x, y = move[0], move[1]
            else:
                continue
                
            for dx in range(-3, 4):  # Expanded from -2:3 to -3:4
                for dy in range(-3, 4):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.board.rows and 0 <= new_y < self.board.cols and self.board.grid[new_x][new_y] == 0:
                        candidate_move.add((new_x, new_y))

        return list(candidate_move)
    
    def score_move(self, x: int, y: int, player: int):
        """Score a move based on minimax evaluation"""
        # Clear killer moves for fresh search
        self.killer_moves = [set() for _ in range(self.max_depth + 2)]
        self.evaluation_count = 0
        
        # Temporarily place the piece on the board
        self.board.grid[x][y] = player
        self.board.move_history.append((x, y, player))
        prev_turn = self.board.turn
        self.board.turn = -player

        score = self.minimax(self.max_depth - 1, False, -math.inf, math.inf, player)

        # Undo the move
        self.board.move_history.pop()
        self.board.grid[x][y] = 0
        self.board.turn = prev_turn

        return score

    def minimax(self, depth: int, is_maximizing: bool, alpha: int, beta: int, player: int):
        """Minimax algorithm with alpha-beta pruning and enhancements"""
        
        # Check terminal states first
        winner = self.board.get_winner()
        if winner == player:  # Current player wins
            return 100000 + (self.max_depth - depth) * 10  # Bonus for quick wins
        elif winner == -player:  # Opponent wins
            return -100000 - (self.max_depth - depth) * 10  # Penalty for quick losses
        elif winner == 2:  # Draw
            return 0
        
        if depth == 0:
            self.evaluation_count += 1
            return self.get_heuristic(player)
        
        valid_moves = self.get_candidate_moves()
        if not valid_moves:
            return 0
        
        # Aggressive move ordering: killer moves + heuristic
        sorted_moves = self._order_moves(valid_moves, depth, is_maximizing)
        # Limit branching factor based on depth
        branch_limit = 12 if depth > 2 else 15
        sorted_moves = sorted_moves[:branch_limit]
        
        if is_maximizing:
            max_eval = -math.inf
            best_move = None
            for move in sorted_moves:
                x, y = move
                self.board.grid[x][y] = player
                self.board.move_history.append((x, y, player))
                prev_turn = self.board.turn
                self.board.turn = -player

                eval_score = self.minimax(depth - 1, False, alpha, beta, player)

                self.board.move_history.pop()
                self.board.grid[x][y] = 0
                self.board.turn = prev_turn

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    # Update killer moves for good move
                    if depth <= len(self.killer_moves) - 1:
                        self.killer_moves[depth].add(move)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cut-off

            return max_eval
        else:
            min_eval = math.inf
            best_move = None
            for move in sorted_moves:
                x, y = move
                self.board.grid[x][y] = -player
                self.board.move_history.append((x, y, -player))
                prev_turn = self.board.turn
                self.board.turn = player

                eval_score = self.minimax(depth - 1, True, alpha, beta, player)

                self.board.move_history.pop()
                self.board.grid[x][y] = 0
                self.board.turn = prev_turn

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    # Update killer moves for good move
                    if depth <= len(self.killer_moves) - 1:
                        self.killer_moves[depth].add(move)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cut-off

            return min_eval
        
    def get_heuristic(self, player: int):
        """ Get a heuristic score for the current board state """
        return self._heuristic()
    
    def _heuristic(self):
        """Calculate comprehensive heuristic score for board state"""
        score = 0
        score += self._evaluate_lines(True)   # Horizontal
        score += self._evaluate_lines(False)  # Vertical
        score += self._evaluate_diagonals()   # Diagonals
        return score
    
    def _evaluate_lines(self, is_horizontal):
        """Evaluate horizontal or vertical lines"""
        score = 0
        for i in range(self.board.rows):
            for j in range(self.board.cols - 4):
                if is_horizontal:
                    window = self.board.grid[i][j:j + 5]
                else:
                    window = [self.board.grid[j + k][i] for k in range(5)]
                score += self._evaluate_window(window)
        return score
    
    def _evaluate_diagonals(self):
        """Evaluate diagonal lines"""
        score = 0
        
        # Top-left to bottom-right diagonals
        for i in range(self.board.rows - 4):
            for j in range(self.board.cols - 4):
                diagonal = [self.board.grid[i + k][j + k] for k in range(5)]
                score += self._evaluate_window(diagonal)
        
        # Top-right to bottom-left diagonals
        for i in range(self.board.rows - 4):
            for j in range(4, self.board.cols):
                diagonal = [self.board.grid[i + k][j - k] for k in range(5)]
                score += self._evaluate_window(diagonal)
        
        return score
    
    def _evaluate_window(self, window):
        """Enhanced window evaluation with better heuristic values"""
        ai_pattern = self._get_line_pattern(window, -1)  # -1 for AI
        player_pattern = self._get_line_pattern(window, 1)  # 1 for Player
        
        ai_count, ai_open = ai_pattern
        player_count, player_open = player_pattern
        
        # AI patterns (maximize) - Improved scoring
        if ai_count == 5:
            return 100000  # Win
        if ai_count == 4:
            # 4 in a row - critical situation
            if ai_open >= 1:
                return 50000  # Can win next move
            else:
                return 8000  # Blocked but still valuable
        if ai_count == 3:
            # 3 in a row - building threat
            if ai_open == 2:
                return 3000  # Open on both sides - very strong
            elif ai_open == 1:
                return 800  # Semi-open
            else:
                return 100  # Blocked
        if ai_count == 2:
            # 2 in a row - potential
            if ai_open == 2:
                return 200  # Open both sides
            elif ai_open == 1:
                return 30
        if ai_count == 1:
            if ai_open == 2:
                return 5  # Single piece with space
        
        # Player patterns (minimize) - Strong defense
        if player_count == 5:
            return -100000  # Loss - critical
        if player_count == 4:
            # Must block immediately
            if player_open >= 1:
                return -50000  # Opponent can win next - URGENT
            else:
                return -8000  # Blocked threat
        if player_count == 3:
            # 3 in a row - must watch
            if player_open == 2:
                return -5000  # Very dangerous - increase penalty
            elif player_open == 1:
                return -1500  # Semi-threat
            else:
                return -200  # Blocked
        if player_count == 2:
            # 2 in a row
            if player_open == 2:
                return -400  # Open both sides
            elif player_open == 1:
                return -60
        if player_count == 1:
            if player_open == 2:
                return -8
        
        return 0
    
    def _get_line_pattern(self, window, player):
        """Analyze pattern in 5-cell window"""
        count = 0
        open_ends = 0
        
        for cell in window:
            if cell == player:
                count += 1
            elif cell != 0:
                return (0, 0)  # Blocked
        
        if count == 0:
            return (0, 0)
        
        # Count open ends
        if window[0] == 0:
            open_ends += 1
        if window[4] == 0:
            open_ends += 1
        
        return (count, open_ends)
    
    def _evaluate_move(self, row, col):
        """Improved quick move evaluation for better ordering"""
        score_ai = 0
        score_player = 0
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # Check AI piece count
            count_ai = 1
            for i in range(1, 5):
                nx, ny = row + i * dx, col + i * dy
                if 0 <= nx < self.board.rows and 0 <= ny < self.board.cols and self.board.grid[nx][ny] == -1:
                    count_ai += 1
                else:
                    break
            for i in range(1, 5):
                nx, ny = row - i * dx, col - i * dy
                if 0 <= nx < self.board.rows and 0 <= ny < self.board.cols and self.board.grid[nx][ny] == -1:
                    count_ai += 1
                else:
                    break
            
            # Check Player piece count
            count_player = 1
            for i in range(1, 5):
                nx, ny = row + i * dx, col + i * dy
                if 0 <= nx < self.board.rows and 0 <= ny < self.board.cols and self.board.grid[nx][ny] == 1:
                    count_player += 1
                else:
                    break
            for i in range(1, 5):
                nx, ny = row - i * dx, col - i * dy
                if 0 <= nx < self.board.rows and 0 <= ny < self.board.cols and self.board.grid[nx][ny] == 1:
                    count_player += 1
                else:
                    break
            
            # Improved scoring
            if count_ai >= 4:
                score_ai += 100000  # Critical AI threat
            elif count_ai == 3:
                score_ai += 5000
            elif count_ai == 2:
                score_ai += 300
            
            if count_player >= 4:
                score_player += 100000  # Critical player threat - must block
            elif count_player == 3:
                score_player += 5000  # Player threat
            elif count_player == 2:
                score_player += 300
        
        # Consider center proximity for early game strategies
        center = self.board.rows // 2
        distance_from_center = abs(row - center) + abs(col - center)
        center_bonus = max(0, 50 - distance_from_center)  # Prefer center
        
        return score_ai + score_player * 1.5 + center_bonus  # Strong defense priority
    
    def _sort_moves_by_heuristic(self, moves):
        """Sort moves by heuristic evaluation (best first)"""
        move_scores = []
        for x, y in moves:
            score = self._evaluate_move(x, y)
            move_scores.append((score, x, y))
        
        move_scores.sort(reverse=True)
        return [(x, y) for score, x, y in move_scores]
    
    def _order_moves(self, moves, depth, is_maximizing):
        """Order moves using heuristics: killer moves + evaluation"""
        # Start with killer moves if available
        ordered = []
        killer_set = self.killer_moves[depth] if depth < len(self.killer_moves) else set()
        
        # Add killer moves first
        killer_list = [(m[0] * 1000000 + m[1], m) for m in killer_set if m in moves]
        killer_list.sort(reverse=True)
        for _, move in killer_list:
            ordered.append(move)
        
        # Add remaining moves sorted by heuristic
        remaining = [m for m in moves if m not in killer_set]
        other_scores = []
        for x, y in remaining:
            score = self._evaluate_move(x, y)
            other_scores.append((score, x, y))
        
        other_scores.sort(reverse=True)
        for _, x, y in other_scores:
            ordered.append((x, y))
        
        return ordered
    
    def get_move(self):
        """Get the best move using optimized minimax with killer moves"""
        valid_moves = self.get_candidate_moves()
        if not valid_moves:
            return None
        
        best_score = -math.inf
        best_move = None

        # Sort moves by heuristic for better alpha-beta pruning
        sorted_moves = self._sort_moves_by_heuristic(valid_moves)
        
        # Evaluate top candidates
        num_candidates = 10 if len(sorted_moves) > 12 else 12
        for move in sorted_moves[:num_candidates]:
            x, y = move
            score = self.score_move(x, y, player=self.board.turn)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
