import os
import numpy as np
import torch

from game.board import Board
from neural_net.architecture import GomokuNet
from mcts.mcts_alpha_zero import MCTS


class AgentRL:
    """
    Reinforcement Learning agent that uses AlphaZero-style MCTS + Neural Network.

    This agent can be used as a drop-in replacement for AgentMiniMax in the game.
    It loads a trained neural network checkpoint and uses MCTS to select moves.
    """

    def __init__(self, board: Board, checkpoint_path: str = None,
                 num_simulations: int = 200, device: str = None):
        """
        Args:
            board: Board object from the game
            checkpoint_path: path to model checkpoint (.pth file)
            num_simulations: number of MCTS simulations per move (higher = stronger but slower)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.board = board

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize neural network
        self.net = GomokuNet(
            board_size=board.rows,
            in_channels=4,
            num_res_blocks=5,
            channels=128,
        )
        self.net.to(self.device)



        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading RL agent from checkpoint: {checkpoint_path}")
            self.net.load_checkpoint(checkpoint_path, device=self.device)
        else:
            print("No checkpoint path provided or file does not exist. Attempting to load default checkpoint...")
            # Use default checkpoint path
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'neural_net', 'model_checkpoint.pth'
            )
            if os.path.exists(default_path):
                print(f"Loading default checkpoint: {default_path}")
                self.net.load_checkpoint(default_path, device=self.device)

        self.net.eval()

        # Initialize MCTS
        self.mcts = MCTS(
            neural_net=self.net,
            num_simulations=num_simulations,
            c_puct=1.5,
        )

    def get_move(self):
        """
        Get the best move for the current board state.
        Compatible with the same interface as AgentMiniMax.get_move().

        Returns:
            (row, col) tuple or None if no valid moves
        """
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None

        # Use greedy selection (temperature=0) for actual gameplay
        action, _ = self.mcts.get_action(self.board, temperature=0)
        return action

    def get_move_with_probs(self, temperature=1.0):
        """
        Get move along with the MCTS policy distribution.
        Used during self-play for training data collection.

        Args:
            temperature: controls exploration (1.0 = proportional, 0 = greedy)

        Returns:
            (row, col), action_probs
        """
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None, np.zeros(self.board.rows * self.board.cols)

        action, action_probs = self.mcts.get_action(self.board, temperature=temperature)
        return action, action_probs
