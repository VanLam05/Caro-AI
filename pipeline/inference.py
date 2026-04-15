import torch
from game.board import Board
from models.agentRL import AgentRL
from models.agentMiniMax import AgentMiniMax

class InferenceAgent:
    """Use trained RL model for inference/playing"""
    
    def __init__(self, board: Board, model_path: str):
        self.board = board
        self.rl_agent = AgentRL(board)
        self.rl_agent.load_model(model_path)
        self.rl_agent.epsilon = 0  # No exploration during inference
    
    def get_best_move(self, player: int):
        """
        Get the best move according to the trained model
        Args:
            player: Current player (1 or -1)
        
        Returns:
            Tuple of (x, y) coordinates
        """
        return self.rl_agent.select_move(player, use_epsilon=False)
    
    def play_vs_minimax(self, minimax_depth: int = 3, rl_first: bool = True):
        """
        Play one game between trained RL agent and MiniMax
        Args:
            minimax_depth: Depth of MiniMax search
            rl_first: If True, RL plays first
        
        Returns:
            (winner, move_history): winner is 1 (RL wins), -1 (MiniMax wins), 0 (draw)
        """
        self.board.reset()
        
        rl_player = 1 if rl_first else -1
        minimax_player = -rl_player
        minimax_agent = AgentMiniMax(self.board, max_depth=minimax_depth)
        
        move_history = []
        max_moves = self.board.rows * self.board.cols
        move_count = 0
        
        while move_count < max_moves:
            # RL Agent's turn
            if self.board.turn == rl_player:
                move = self.get_best_move(rl_player)
                
                if move is None:
                    break
                
                x, y = move
                if not self.board.make_move(x, y):
                    continue
                
                move_history.append((x, y, rl_player, "RL"))
                move_count += 1
                
                print(f"RL Agent ({rl_player:+d}) plays at ({x}, {y})")
            
            # MiniMax Agent's turn
            elif self.board.turn == minimax_player:
                # Find best move
                valid_moves = minimax_agent.get_candidate_moves()
                best_move = None
                best_score = float('-inf')
                
                for m in valid_moves:
                    x, y = m
                    score = minimax_agent.score_move(x, y, minimax_player)
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
                
                if best_move is None:
                    break
                
                x, y = best_move
                if not self.board.make_move(x, y):
                    continue
                
                move_history.append((x, y, minimax_player, "MiniMax"))
                move_count += 1
                
                print(f"MiniMax Agent ({minimax_player:+d}) plays at ({x}, {y})")
            
            # Check winner
            winner = self.board.get_winner()
            if winner != 0:
                return winner, move_history
        
        return 0, move_history  # Draw or inconclusive


def play_games(model_path: str, num_games: int = 5, minimax_depth: int = 3):
    """
    Play multiple games between trained RL agent and MiniMax for evaluation
    
    Args:
        model_path: Path to saved model
        num_games: Number of games to play
        minimax_depth: Depth of MiniMax search
    """
    board = Board(rows=15, cols=15)
    inference = InferenceAgent(board, model_path)
    
    rl_wins = 0
    minimax_wins = 0
    draws = 0
    
    print(f"Playing {num_games} games for evaluation...")
    print("=" * 80)
    
    for game_num in range(1, num_games + 1):
        print(f"\nGame {game_num}/{num_games}")
        print("-" * 80)
        
        # Alternate who goes first
        rl_first = game_num % 2 == 1
        rl_player = 1 if rl_first else -1
        
        print(f"RL Agent plays as: {'Player X (1)' if rl_first else 'Player O (-1)'}")
        print()
        
        winner, move_history = inference.play_vs_minimax(minimax_depth=minimax_depth, rl_first=rl_first)
        
        if winner == rl_player:
            print(f"\n✓ RL Agent wins!")
            rl_wins += 1
        elif winner == -rl_player:
            print(f"\n✗ MiniMax wins!")
            minimax_wins += 1
        else:
            print(f"\n= Draw")
            draws += 1
        
        print(f"Total moves: {len(move_history)}")
    
    # Print statistics
    total_games = rl_wins + minimax_wins + draws
    rl_win_rate = (rl_wins / total_games * 100) if total_games > 0 else 0
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Games: {total_games}")
    print(f"RL Wins: {rl_wins} ({rl_win_rate:.2f}%)")
    print(f"MiniMax Wins: {minimax_wins} ({100 - rl_win_rate:.2f}%)")
    print(f"Draws: {draws}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    MODEL_PATH = "models/rl_model.pth"
    
    # Play 5 games for evaluation
    play_games(MODEL_PATH, num_games=5, minimax_depth=3)
