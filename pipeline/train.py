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
    Two-phase training pipeline with optimized hyperparameters.

    Key optimizations:
      - Dirichlet alpha=0.15 (tuned for 225-action space: ~10/N)
      - c_puct=2.5 (stronger exploration)
      - Phase 1: fewer MCTS sims (faster), more games, label smoothing
      - CosineAnnealingWarmRestarts scheduler
      - Mixed precision (fp16) on GPU
    """

    def __init__(self, board_size=15, device=None,
                 num_res_blocks=5, channels=128,
                 c_puct=2.5,
                 lr=0.001, weight_decay=1e-4,
                 replay_buffer_size=100000,
                 batch_size=256, epochs_per_iter=10,
                 # Phase 1: vs MiniMax
                 phase1_iterations=20,
                 phase1_games=40,
                 phase1_simulations=100,
                 # Phase 2: Self-play
                 phase2_iterations=15,
                 phase2_games=30,
                 phase2_simulations=200,
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
        self.c_puct = c_puct

        self.phase1_iterations = phase1_iterations
        self.phase1_games = phase1_games
        self.phase1_simulations = phase1_simulations
        self.phase2_iterations = phase2_iterations
        self.phase2_games = phase2_games
        self.phase2_simulations = phase2_simulations

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay,
        )

        total_iters = phase1_iterations + phase2_iterations
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(5, total_iters // 3), T_mult=2,
        )

        # Mixed precision scaler for GPU
        self.use_amp = (self.device == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        """Run the full two-phase training loop."""
        total_iterations = self.phase1_iterations + self.phase2_iterations

        print(f"{'='*60}")
        print(f"Two-Phase Training for Gomoku {self.board_size}x{self.board_size}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Network: {sum(p.numel() for p in self.net.parameters()):,} params")
        print(f"Phase 1: {self.phase1_iterations} iters vs MiniMax "
              f"({self.phase1_games} games, {self.phase1_simulations} sims)")
        print(f"Phase 2: {self.phase2_iterations} iters self-play "
              f"({self.phase2_games} games, {self.phase2_simulations} sims)")
        print(f"LR: {self.lr}, Batch: {self.batch_size}, "
              f"Epochs/iter: {self.epochs_per_iter}")
        print(f"Replay buffer: {self.replay_buffer.maxlen}")
        print(f"{'='*60}")

        total_start = time.time()

        # ============================================================
        # PHASE 1: Learn from MiniMax
        # ============================================================
        print(f"\n{'#'*60}")
        print(f"# PHASE 1: Learning from MiniMax")
        print(f"{'#'*60}")

        phase1_schedule = self._build_minimax_schedule()

        for iteration in range(1, self.phase1_iterations + 1):
            iter_start = time.time()
            minimax_depth = phase1_schedule[iteration - 1]

            print(f"\n{'='*40}")
            print(f"Phase 1 - Iter {iteration}/{self.phase1_iterations} "
                  f"(MiniMax depth={minimax_depth})")
            print(f"{'='*40}")

            print(f"\n[Data] {self.phase1_games} games vs "
                  f"MiniMax (depth={minimax_depth}, "
                  f"sims={self.phase1_simulations})...")
            self.net.eval()
            new_data = collect_vs_minimax_data(
                self.net,
                num_games=self.phase1_games,
                minimax_depth=minimax_depth,
                board_size=self.board_size,
                num_simulations=self.phase1_simulations,
                verbose=True,
            )
            self.replay_buffer.extend(new_data)
            print(f"  +{len(new_data)} examples, "
                  f"buffer: {len(self.replay_buffer)}")

            print("\n[Train] Updating network...")
            train_loss = self._train_network()
            print(f"  Loss: {train_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self._save_checkpoint(iteration, 'phase1')

            if iteration % 5 == 0:
                self._run_evaluation()

            self.scheduler.step()
            self._print_time_stats(iter_start, total_start, iteration,
                                   total_iterations)

        # ============================================================
        # PHASE 2: Self-play refinement
        # ============================================================
        print(f"\n{'#'*60}")
        print(f"# PHASE 2: Self-play Refinement")
        print(f"{'#'*60}")

        # Lower LR for fine-tuning
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr * 0.3,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.phase2_iterations,
        )

        for iteration in range(1, self.phase2_iterations + 1):
            iter_start = time.time()
            global_iter = self.phase1_iterations + iteration

            print(f"\n{'='*40}")
            print(f"Phase 2 - Iter {iteration}/{self.phase2_iterations} "
                  f"(Self-play)")
            print(f"{'='*40}")

            print(f"\n[Data] {self.phase2_games} self-play games "
                  f"(sims={self.phase2_simulations})...")
            self.net.eval()
            new_data = collect_self_play_data(
                self.net,
                num_games=self.phase2_games,
                board_size=self.board_size,
                num_simulations=self.phase2_simulations,
                verbose=True,
            )
            self.replay_buffer.extend(new_data)
            print(f"  +{len(new_data)} examples, "
                  f"buffer: {len(self.replay_buffer)}")

            print("\n[Train] Updating network...")
            train_loss = self._train_network()
            print(f"  Loss: {train_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self._save_checkpoint(global_iter, 'phase2')

            if iteration % 5 == 0:
                self._run_evaluation()

            self.scheduler.step()
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
        Progressive MiniMax difficulty schedule.
        Focuses more on depth 2-3 (the most useful learning range).
        """
        n = self.phase1_iterations
        schedule = []

        # First 25%: depth 1 (learn basic patterns)
        n1 = max(1, n // 4)
        schedule.extend([1] * n1)

        # Next 25%: depth 2 (positional play)
        n2 = max(1, n // 4)
        schedule.extend([2] * n2)

        # Next 30%: depth 3 (tactical play)
        n3 = max(1, n * 3 // 10)
        schedule.extend([3] * n3)

        # Last 20%: depth 3-5 (advanced tactics)
        remaining = n - len(schedule)
        for i in range(remaining):
            schedule.append(3 if i < remaining // 2 else 5)

        return schedule[:n]

    def _train_network(self):
        """Train with optional mixed precision (AMP)."""
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

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        log_policy, value = self.net(states)
                        policy_loss = -torch.sum(
                            target_policies * log_policy
                        ) / len(batch)
                        value_loss = nn.MSELoss()(value, target_values)
                        loss = policy_loss + value_loss

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    log_policy, value = self.net(states)
                    policy_loss = -torch.sum(
                        target_policies * log_policy
                    ) / len(batch)
                    value_loss = nn.MSELoss()(value, target_values)
                    loss = policy_loss + value_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _evaluate_vs_minimax(self, num_games=10, minimax_depth=3):
        """Evaluate current network against MiniMax agent."""
        self.net.eval()
        mcts = MCTS(self.net, num_simulations=self.phase2_simulations,
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
        print(f"  Saved: {checkpoint_path}")

        if iteration % 5 == 0:
            iter_path = os.path.join(
                self.checkpoint_dir, f'model_{phase}_iter_{iteration}.pth'
            )
            self.net.save_checkpoint(iter_path)

    def _run_evaluation(self):
        """Run evaluation against MiniMax at multiple depths."""
        print("\n[Eval] vs MiniMax...")
        for depth in [1, 3]:
            win_rate = self._evaluate_vs_minimax(num_games=10,
                                                  minimax_depth=depth)
            print(f"  depth={depth}: {win_rate:.1%}")

    def _print_time_stats(self, iter_start, total_start, current_iter,
                          total_iters):
        """Print timing statistics."""
        iter_time = time.time() - iter_start
        total_time = time.time() - total_start
        avg_iter_time = total_time / current_iter
        remaining = avg_iter_time * (total_iters - current_iter)
        print(f"\n  Time: {iter_time:.0f}s | "
              f"Total: {total_time/3600:.1f}h | "
              f"ETA: {remaining/3600:.1f}h")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Two-Phase Training for Gomoku AI'
    )
    parser.add_argument('--phase1-iterations', type=int, default=20,
                        help='Phase 1 iterations (vs MiniMax, default: 20)')
    parser.add_argument('--phase1-games', type=int, default=40,
                        help='Games per Phase 1 iteration (default: 40)')
    parser.add_argument('--phase1-simulations', type=int, default=100,
                        help='MCTS sims for Phase 1 (default: 100)')
    parser.add_argument('--phase2-iterations', type=int, default=15,
                        help='Phase 2 iterations (self-play, default: 15)')
    parser.add_argument('--phase2-games', type=int, default=30,
                        help='Games per Phase 2 iteration (default: 30)')
    parser.add_argument('--phase2-simulations', type=int, default=200,
                        help='MCTS sims for Phase 2 (default: 200)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
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
        phase1_simulations=args.phase1_simulations,
        phase2_iterations=args.phase2_iterations,
        phase2_games=args.phase2_games,
        phase2_simulations=args.phase2_simulations,
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
