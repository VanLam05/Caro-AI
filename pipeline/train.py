"""
Two-phase training pipeline for Gomoku AI.

Phase 1 - Learn from MiniMax:
    Play against MiniMax agents at increasing depths.
    The network learns from MiniMax's expert moves (imitation)
    and game outcomes (value prediction). This bootstraps the
    network much faster than pure self-play.

Phase 2 - Self-play refinement:
    Once the network is strong enough, switch to self-play
    to push beyond MiniMax's level.

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
from pipeline.collect_data import collect_vs_minimax_data, collect_self_play_data
from game.board import Board
from models.agentMiniMax import AgentMiniMax


class AlphaZeroTrainer:
    """
    Two-phase training pipeline:
      Phase 1: Train against MiniMax (depth 1 → 3 → 5) to learn basics quickly
      Phase 2: Self-play to surpass MiniMax level
    """

    def __init__(self, board_size=15, device=None,
                 num_res_blocks=5, channels=128,
                 num_simulations=200, c_puct=1.5,
                 lr=0.002, weight_decay=1e-4,
                 replay_buffer_size=50000,
                 batch_size=256, epochs_per_iter=5,
                 # Phase 1: vs MiniMax
                 phase1_iterations=20,
                 phase1_games=30,
                 # Phase 2: Self-play
                 phase2_iterations=15,
                 phase2_games=25,
                 checkpoint_dir='neural_net'):
        self.board_size = board_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = GomokuNet(
            board_size=board_size,
            in_channels=4,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        self.phase1_iterations = phase1_iterations
        self.phase1_games = phase1_games
        self.phase2_iterations = phase2_iterations
        self.phase2_games = phase2_games

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )

    def train(self):
        """Run the full two-phase training loop."""
        total_iterations = self.phase1_iterations + self.phase2_iterations

        print(f"{'='*60}")
        print(f"Two-Phase Training for Gomoku {self.board_size}x{self.board_size}")
        print(f"Device: {self.device}")
        print(f"Network: {sum(p.numel() for p in self.net.parameters()):,} params")
        print(f"Phase 1: {self.phase1_iterations} iterations vs MiniMax "
              f"({self.phase1_games} games/iter)")
        print(f"Phase 2: {self.phase2_iterations} iterations self-play "
              f"({self.phase2_games} games/iter)")
        print(f"{'='*60}")

        total_start = time.time()

        # ============================================================
        # PHASE 1: Learn from MiniMax
        # ============================================================
        print(f"\n{'#'*60}")
        print(f"# PHASE 1: Learning from MiniMax")
        print(f"{'#'*60}")

        # Progressive difficulty: start easy, increase depth
        phase1_schedule = self._build_minimax_schedule()

        for iteration in range(1, self.phase1_iterations + 1):
            iter_start = time.time()
            minimax_depth = phase1_schedule[iteration - 1]

            print(f"\n{'='*40}")
            print(f"Phase 1 - Iteration {iteration}/{self.phase1_iterations} "
                  f"(MiniMax depth={minimax_depth})")
            print(f"{'='*40}")

            # Collect data vs MiniMax
            print(f"\n[Data] Playing {self.phase1_games} games vs "
                  f"MiniMax (depth={minimax_depth})...")
            self.net.eval()
            new_data = collect_vs_minimax_data(
                self.net,
                num_games=self.phase1_games,
                minimax_depth=minimax_depth,
                board_size=self.board_size,
                num_simulations=self.num_simulations,
                verbose=True,
            )
            self.replay_buffer.extend(new_data)
            print(f"  New examples: {len(new_data)}, "
                  f"Buffer: {len(self.replay_buffer)}")

            # Train
            print("\n[Train] Updating neural network...")
            train_loss = self._train_network()
            print(f"  Loss: {train_loss:.4f}")

            # Save
            self._save_checkpoint(iteration, 'phase1')

            # Evaluate every 5 iterations
            if iteration % 5 == 0:
                self._run_evaluation(iteration, total_iterations)

            self.scheduler.step()
            self._print_time_stats(iter_start, total_start, iteration,
                                   total_iterations)

        # ============================================================
        # PHASE 2: Self-play refinement
        # ============================================================
        print(f"\n{'#'*60}")
        print(f"# PHASE 2: Self-play Refinement")
        print(f"{'#'*60}")

        # Reset optimizer with lower LR for fine-tuning
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr * 0.5,
            weight_decay=self.weight_decay,
        )

        for iteration in range(1, self.phase2_iterations + 1):
            iter_start = time.time()
            global_iter = self.phase1_iterations + iteration

            print(f"\n{'='*40}")
            print(f"Phase 2 - Iteration {iteration}/{self.phase2_iterations} "
                  f"(Self-play)")
            print(f"{'='*40}")

            # Self-play
            print(f"\n[Data] Playing {self.phase2_games} self-play games...")
            self.net.eval()
            new_data = collect_self_play_data(
                self.net,
                num_games=self.phase2_games,
                board_size=self.board_size,
                num_simulations=self.num_simulations,
                verbose=True,
            )
            self.replay_buffer.extend(new_data)
            print(f"  New examples: {len(new_data)}, "
                  f"Buffer: {len(self.replay_buffer)}")

            # Train
            print("\n[Train] Updating neural network...")
            train_loss = self._train_network()
            print(f"  Loss: {train_loss:.4f}")

            # Save
            self._save_checkpoint(global_iter, 'phase2')

            # Evaluate every 5 iterations
            if iteration % 5 == 0:
                self._run_evaluation(global_iter, total_iterations)

            self._print_time_stats(iter_start, total_start, global_iter,
                                   total_iterations)

        # ============================================================
        # Final evaluation
        # ============================================================
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        for depth, n_games in [(1, 20), (3, 20), (5, 10)]:
            win_rate = self._evaluate_vs_minimax(num_games=n_games,
                                                  minimax_depth=depth)
            print(f"  vs MiniMax depth={depth}: {win_rate:.1%} "
                  f"({int(win_rate * n_games)}/{n_games})")

        total_time = time.time() - total_start
        print(f"\nTotal training time: {total_time/3600:.2f} hours")

        final_path = os.path.join(self.checkpoint_dir, 'model_checkpoint.pth')
        self.net.save_checkpoint(final_path)
        print(f"Final model saved: {final_path}")

    def _build_minimax_schedule(self):
        """
        Build progressive MiniMax difficulty schedule.
        Starts with depth=1, gradually increases to depth=3, then depth=5.
        """
        n = self.phase1_iterations
        schedule = []

        # First 30%: depth 1 (easy)
        n1 = max(1, n * 3 // 10)
        schedule.extend([1] * n1)

        # Next 40%: depth 2-3 (medium)
        n2 = max(1, n * 4 // 10)
        for i in range(n2):
            schedule.append(2 if i < n2 // 2 else 3)

        # Last 30%: depth 3-5 (hard)
        remaining = n - len(schedule)
        for i in range(remaining):
            schedule.append(3 if i < remaining // 2 else 5)

        return schedule[:n]

    def _train_network(self):
        """Train the neural network on replay buffer data."""
        self.net.train()

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs_per_iter):
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

                log_policy, value = self.net(states)

                policy_loss = -torch.sum(target_policies * log_policy) / len(batch)
                value_loss = nn.MSELoss()(value, target_values)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _evaluate_vs_minimax(self, num_games=10, minimax_depth=3):
        """Evaluate current network against MiniMax agent."""
        self.net.eval()
        mcts = MCTS(self.net, num_simulations=self.num_simulations,
                     c_puct=self.c_puct)

        wins = 0
        draws = 0

        for game_idx in range(num_games):
            board = Board(rows=self.board_size, cols=self.board_size,
                          winning_condition=5)
            minimax = AgentMiniMax(board, max_depth=minimax_depth)

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

    def _save_checkpoint(self, iteration, phase):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'
        )
        self.net.save_checkpoint(checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        if iteration % 5 == 0:
            iter_path = os.path.join(
                self.checkpoint_dir, f'model_{phase}_iter_{iteration}.pth'
            )
            self.net.save_checkpoint(iter_path)

    def _run_evaluation(self, current_iter, total_iters):
        """Run evaluation against MiniMax at multiple depths."""
        print("\n[Eval] Evaluating vs MiniMax...")
        for depth in [1, 3]:
            win_rate = self._evaluate_vs_minimax(num_games=10,
                                                  minimax_depth=depth)
            print(f"  vs depth={depth}: {win_rate:.1%}")

    def _print_time_stats(self, iter_start, total_start, current_iter,
                          total_iters):
        """Print timing statistics."""
        iter_time = time.time() - iter_start
        total_time = time.time() - total_start
        avg_iter_time = total_time / current_iter
        remaining = avg_iter_time * (total_iters - current_iter)
        print(f"\n  Time: {iter_time:.0f}s this iter, "
              f"{total_time/3600:.1f}h total, "
              f"~{remaining/3600:.1f}h remaining")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Two-Phase Training for Gomoku AI'
    )
    parser.add_argument('--phase1-iterations', type=int, default=20,
                        help='Phase 1 iterations (vs MiniMax, default: 20)')
    parser.add_argument('--phase1-games', type=int, default=30,
                        help='Games per Phase 1 iteration (default: 30)')
    parser.add_argument('--phase2-iterations', type=int, default=15,
                        help='Phase 2 iterations (self-play, default: 15)')
    parser.add_argument('--phase2-games', type=int, default=25,
                        help='Games per Phase 2 iteration (default: 25)')
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
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 (resume directly to self-play)')

    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        phase1_iterations=args.phase1_iterations,
        phase1_games=args.phase1_games,
        phase2_iterations=args.phase2_iterations,
        phase2_games=args.phase2_games,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.net.load_checkpoint(args.resume, device=trainer.device)

    if args.skip_phase1:
        print("Skipping Phase 1 (vs MiniMax), starting Phase 2 (self-play)")
        trainer.phase1_iterations = 0

    trainer.train()
