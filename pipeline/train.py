"""
AlphaZero-style training pipeline for Gomoku AI.

Training loop:
    1. Self-play: Generate games using current network + MCTS
    2. Train: Update network on collected data
    3. Evaluate: Compare new network against previous version / MiniMax agent
    4. Repeat

Designed to run on Google Colab (GPU) in < 1 day.
"""

import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_net.architecture import GomokuNet
from mcts.mcts_alpha_zero import MCTS
from pipeline.collect_data import collect_self_play_data
from game.board import Board
from models.agentMiniMax import AgentMiniMax


class AlphaZeroTrainer:
    """
    Full AlphaZero training pipeline.

    Hyperparameters are tuned for Colab feasibility:
    - Smaller network (5 res blocks, 128 channels)
    - Fewer MCTS simulations (200 per move)
    - Shorter training iterations
    """

    def __init__(self, board_size=15, device=None,
                 num_res_blocks=5, channels=128,
                 num_simulations=200, c_puct=1.5,
                 lr=0.002, weight_decay=1e-4,
                 replay_buffer_size=50000,
                 batch_size=256, epochs_per_iter=5,
                 self_play_games=25, num_iterations=30,
                 checkpoint_dir='neural_net'):
        """
        Args:
            board_size: board dimensions (15 for standard Gomoku)
            device: 'cuda' or 'cpu'
            num_res_blocks: residual blocks in the network
            channels: convolutional channels
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
            lr: learning rate
            weight_decay: L2 regularization
            replay_buffer_size: max training examples to keep
            batch_size: training batch size
            epochs_per_iter: training epochs per iteration
            self_play_games: games per self-play phase
            num_iterations: total training iterations
            checkpoint_dir: directory for model checkpoints
        """
        self.board_size = board_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Network
        self.net = GomokuNet(
            board_size=board_size,
            in_channels=4,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(self.device)

        # Training params
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.self_play_games = self_play_games
        self.num_iterations = num_iterations
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )

    def train(self):
        """Run the full training loop."""
        print(f"{'='*60}")
        print(f"AlphaZero Training for Gomoku {self.board_size}x{self.board_size}")
        print(f"Device: {self.device}")
        print(f"Network: {self.net.__class__.__name__} "
              f"({sum(p.numel() for p in self.net.parameters()):,} params)")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Self-play games per iteration: {self.self_play_games}")
        print(f"Training iterations: {self.num_iterations}")
        print(f"{'='*60}")

        total_start = time.time()

        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()
            print(f"\n{'='*40}")
            print(f"Iteration {iteration}/{self.num_iterations}")
            print(f"{'='*40}")

            # Phase 1: Self-play
            print("\n[Phase 1] Self-play data collection...")
            self.net.eval()
            new_data = collect_self_play_data(
                self.net,
                num_games=self.self_play_games,
                board_size=self.board_size,
                num_simulations=self.num_simulations,
                verbose=True,
            )
            self.replay_buffer.extend(new_data)
            print(f"  New examples: {len(new_data)}, "
                  f"Buffer size: {len(self.replay_buffer)}")

            # Phase 2: Training
            print("\n[Phase 2] Neural network training...")
            train_loss = self._train_network()
            print(f"  Average loss: {train_loss:.4f}")

            # Phase 3: Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 'model_checkpoint.pth'
            )
            self.net.save_checkpoint(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            # Also save iteration-specific checkpoint every 5 iterations
            if iteration % 5 == 0:
                iter_path = os.path.join(
                    self.checkpoint_dir, f'model_iter_{iteration}.pth'
                )
                self.net.save_checkpoint(iter_path)

            # Phase 4: Evaluation (every 5 iterations)
            if iteration % 5 == 0:
                print("\n[Phase 3] Evaluation vs MiniMax...")
                win_rate = self._evaluate_vs_minimax(num_games=10)
                print(f"  Win rate vs MiniMax (depth=3): {win_rate:.1%}")

            # Step the LR scheduler
            self.scheduler.step()

            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            print(f"\n  Iteration time: {iter_time:.0f}s, "
                  f"Total: {total_time/3600:.1f}h")

            # Estimate remaining time
            avg_iter_time = total_time / iteration
            remaining = avg_iter_time * (self.num_iterations - iteration)
            print(f"  Estimated remaining: {remaining/3600:.1f}h")

        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        win_rate_easy = self._evaluate_vs_minimax(num_games=20, minimax_depth=1)
        win_rate_medium = self._evaluate_vs_minimax(num_games=20, minimax_depth=3)
        print(f"  vs MiniMax depth=1 (easy):   {win_rate_easy:.1%}")
        print(f"  vs MiniMax depth=3 (medium): {win_rate_medium:.1%}")

        total_time = time.time() - total_start
        print(f"\nTotal training time: {total_time/3600:.2f} hours")

        # Final save
        final_path = os.path.join(self.checkpoint_dir, 'model_checkpoint.pth')
        self.net.save_checkpoint(final_path)
        print(f"Final model saved: {final_path}")

    def _train_network(self):
        """Train the neural network on replay buffer data."""
        self.net.train()

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs_per_iter):
            # Shuffle data
            data = list(self.replay_buffer)
            random.shuffle(data)

            for batch_start in range(0, len(data), self.batch_size):
                batch = data[batch_start:batch_start + self.batch_size]
                if len(batch) < self.batch_size // 2:
                    continue

                states = torch.FloatTensor(
                    np.array([d[0] for d in batch])
                ).to(self.device)
                target_policies = torch.FloatTensor(
                    np.array([d[1] for d in batch])
                ).to(self.device)
                target_values = torch.FloatTensor(
                    np.array([d[2] for d in batch])
                ).unsqueeze(1).to(self.device)

                # Forward pass
                log_policy, value = self.net(states)

                # Policy loss (cross-entropy)
                policy_loss = -torch.sum(target_policies * log_policy) / len(batch)

                # Value loss (MSE)
                value_loss = nn.MSELoss()(value, target_values)

                # Total loss
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _evaluate_vs_minimax(self, num_games=10, minimax_depth=3):
        """
        Evaluate current network against MiniMax agent.

        Returns:
            win_rate: fraction of games won by RL agent
        """
        self.net.eval()
        mcts = MCTS(self.net, num_simulations=self.num_simulations, c_puct=self.c_puct)

        wins = 0
        draws = 0

        for game_idx in range(num_games):
            board = Board(rows=self.board_size, cols=self.board_size, winning_condition=5)
            minimax = AgentMiniMax(board, max_depth=minimax_depth)

            # Alternate who goes first
            rl_player = 1 if game_idx % 2 == 0 else -1

            move_count = 0
            while True:
                if board.turn == rl_player:
                    action, _ = mcts.get_action(board, temperature=0)
                    row, col = action
                else:
                    move = minimax.get_move()
                    if move is None:
                        break
                    row, col = move

                board.make_move(row, col)
                move_count += 1

                result = board.get_game_ended()
                if result != 0:
                    if result == 0.5:
                        draws += 1
                    elif (result == 1 and rl_player == board.originXO) or \
                         (result == -1 and rl_player != board.originXO):
                        wins += 1
                    break

                if move_count >= board.rows * board.cols:
                    draws += 1
                    break

        return wins / num_games


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train AlphaZero Gomoku Agent')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of training iterations (default: 30)')
    parser.add_argument('--games', type=int, default=25,
                        help='Self-play games per iteration (default: 25)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    parser.add_argument('--checkpoint-dir', type=str, default='neural_net',
                        help='Checkpoint directory (default: neural_net)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        num_iterations=args.iterations,
        self_play_games=args.games,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.net.load_checkpoint(args.resume, device=trainer.device)

    trainer.train()
