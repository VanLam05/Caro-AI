import os
import torch
import numpy as np
from game.board import Board
from models.agentRL import AgentRL
from models.agentMiniMax import AgentMiniMax

class Trainer:
    """Trainer for RL Agent using self-play with MiniMax"""
    
    def __init__(self, num_episodes: int = 100, model_save_path: str = "models/rl_model.pth"):
        self.num_episodes = num_episodes
        self.model_save_path = model_save_path
        self.board = Board(rows=15, cols=15)
        self.rl_agent = AgentRL(self.board)
        self.minimax_agent = AgentMiniMax(self.board, max_depth=3)
        
        # Statistics
        self.rl_wins = 0
        self.minimax_wins = 0
        self.draws = 0
    
    def play_game(self, rl_first: bool = True):
        """
        Play one game between RL agent and MiniMax agent
        Args:
            rl_first: If True, RL agent plays first (value 1), else it plays second (value -1)
        
        Returns:
            winner: 1 if RL wins, -1 if MiniMax wins, 0 if draw, 2 if game didn't complete
        """
        self.board.reset()
        
        rl_player = 1 if rl_first else -1
        minimax_player = -rl_player
        
        episode_memory = []
        move_count = 0
        max_moves = self.board.rows * self.board.cols
        
        while move_count < max_moves:
            # RL Agent's turn
            if self.board.turn == rl_player:
                state_before = self.rl_agent.board_to_tensor(self.board).clone()
                move = self.rl_agent.select_move(rl_player, use_epsilon=True)
                
                if move is None:
                    break
                
                x, y = move
                if not self.board.make_move(x, y):
                    continue  # Invalid move
                
                move_count += 1
                
                # Store state and action for training
                action_idx = x * self.board.rows + y
                episode_memory.append({
                    'state': state_before,
                    'action': action_idx,
                    'player': rl_player
                })
            
            # MiniMax Agent's turn
            elif self.board.turn == minimax_player:
                # Find best move
                valid_moves = self.minimax_agent.get_candidate_moves()
                best_move = None
                best_score = float('-inf')
                
                for m in valid_moves:
                    x, y = m
                    score = self.minimax_agent.score_move(x, y, minimax_player)
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
                
                if best_move is None:
                    break
                
                x, y = best_move
                if not self.board.make_move(x, y):
                    continue
                
                move_count += 1
            
            # Check for winner
            winner = self.board.get_winner()
            if winner != 0:  # Game ended
                # Calculate rewards and store in replay memory
                for exp in episode_memory:
                    reward = self._calculate_reward(winner, exp['player'])
                    next_state = self.rl_agent.board_to_tensor(self.board)
                    done = True
                    
                    self.rl_agent.remember(
                        exp['state'],
                        exp['action'],
                        reward,
                        next_state,
                        done
                    )
                
                return winner
        
        # Game is draw or didn't complete properly
        return 0
    
    def _calculate_reward(self, game_result, player):
        """
        Calculate reward based on game result
        game_result: 1 if player 1 wins, -1 if player -1 wins, 0 if draw
        player: 1 or -1, the player we're calculating reward for
        """
        if game_result == player:
            return 100  # Win
        elif game_result == -player:
            return -100  # Loss
        elif game_result == 2:
            return 0  # Draw
        else:
            return 0  # Game still ongoing
    
    def train(self, target_network_update_freq: int = 5):
        """
        Train the RL agent
        Args:
            target_network_update_freq: Update target network every N episodes
        """
        print(f"Starting training for {self.num_episodes} episodes...")
        print(f"Model will be saved to: {self.model_save_path}")
        print("-" * 80)
        
        for episode in range(1, self.num_episodes + 1):
            # Alternate which player goes first for balanced training
            rl_first = episode % 2 == 1
            
            # Play game
            winner = self.play_game(rl_first=rl_first)
            
            # Update statistics
            rl_player = 1 if rl_first else -1
            if winner == rl_player:
                self.rl_wins += 1
            elif winner == -rl_player:
                self.minimax_wins += 1
            elif winner == 0 or winner == 2:
                self.draws += 1
            
            # Train on replayed experiences
            for _ in range(10):  # Train multiple times per episode
                self.rl_agent.replay()
            
            # Update target network
            if episode % target_network_update_freq == 0:
                self.rl_agent.update_target_network()
            
            # Print statistics every 10 episodes
            if episode % 10 == 0:
                avg_loss = self.rl_agent.get_average_loss()
                total_games = self.rl_wins + self.minimax_wins + self.draws
                rl_win_rate = (self.rl_wins / total_games * 100) if total_games > 0 else 0
                
                print(f"Episode {episode}/{self.num_episodes}")
                print(f"  RL Wins: {self.rl_wins}, MiniMax Wins: {self.minimax_wins}, Draws: {self.draws}")
                print(f"  Win Rate (RL): {rl_win_rate:.2f}%")
                print(f"  Avg Loss: {avg_loss:.6f}")
                print(f"  Epsilon: {self.rl_agent.epsilon:.4f}")
                print(f"  Memory Size: {len(self.rl_agent.memory)}")
                print("-" * 80)
        
        # Save model after training
        self.save_model()
        self.print_final_stats()
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_save_path) if os.path.dirname(self.model_save_path) else ".", exist_ok=True)
        self.rl_agent.save_model(self.model_save_path)
    
    def print_final_stats(self):
        """Print final training statistics"""
        total_games = self.rl_wins + self.minimax_wins + self.draws
        rl_win_rate = (self.rl_wins / total_games * 100) if total_games > 0 else 0
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total Games Played: {total_games}")
        print(f"RL Wins: {self.rl_wins} ({rl_win_rate:.2f}%)")
        print(f"MiniMax Wins: {self.minimax_wins} ({100 - rl_win_rate:.2f}%)")
        print(f"Draws: {self.draws}")
        print(f"Model saved to: {self.model_save_path}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Configure training parameters
    NUM_EPISODES = 100  # Can increase this for better training
    MODEL_PATH = "models/rl_model.pth"
    
    # Start training
    trainer = Trainer(num_episodes=NUM_EPISODES, model_save_path=MODEL_PATH)
    trainer.train(target_network_update_freq=5)
