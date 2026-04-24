import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and batch normalization."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class GomokuNet(nn.Module):
    """
    Dual-headed neural network for AlphaZero-style Gomoku.

    Input: (batch, 4, 15, 15) tensor from Board.get_state_for_nn()
        - Channel 0: current player's pieces
        - Channel 1: opponent's pieces
        - Channel 2: last move position (one-hot)
        - Channel 3: move count (normalized)

    Output:
        - policy: (batch, 225) log-probabilities over all board positions
        - value:  (batch, 1) estimated win probability in [-1, 1]
    """

    def __init__(self, board_size=15, in_channels=4, num_res_blocks=5, channels=128):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size  # 225

        # Initial convolution
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, self.action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.res_blocks(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return F.log_softmax(policy, dim=1), value

    def predict(self, state):
        """
        Predict policy and value for a single board state.

        Args:
            state: numpy array of shape (4, 15, 15)

        Returns:
            policy: numpy array of shape (225,) - probabilities
            value: float in [-1, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            log_policy, value = self.forward(tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value = value.squeeze(0).item()

        return policy, value

    def save_checkpoint(self, filepath):
        """Save model weights to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'board_size': self.board_size,
        }, filepath)

    def load_checkpoint(self, filepath, device='cpu'):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint['state_dict'])
