import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.board import Board

class DQNNetwork(nn.Module):
    """Deep Q-Network for Gomoku AI"""
    
    def __init__(self, board_size: int = 15, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        input_features = board_size * board_size
        output_size = board_size * board_size
        
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AgentRL:
    """Reinforcement Learning Agent for Gomoku"""
    
    def __init__(self, board: Board, learning_rate: float = 0.001, gamma: float = 0.99):
        self.board = board
        self.board_size = board.rows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = DQNNetwork(self.board_size).to(self.device)
        self.target_network = DQNNetwork(self.board_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Statistics
        self.total_loss = 0
        self.update_count = 0
        
    def board_to_tensor(self, board: Board):
        """Convert board state to tensor"""
        # Flatten board into 1D array: 1 for player, -1 for opponent, 0 for empty
        # We'll use the current player's perspective
        state = np.array(board.grid).flatten().astype(np.float32)
        return torch.tensor(state, device=self.device)
    
    def get_valid_moves(self):
        """Get list of valid move indices"""
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board.grid[i][j] == 0:
                    valid_moves.append(i * self.board_size + j)
        return valid_moves
    
    def select_move(self, player: int, use_epsilon: bool = True):
        """
        Select a move using epsilon-greedy strategy
        Args:
            player: Current player (1 or -1)
            use_epsilon: Whether to use epsilon-greedy (True during training, False during inference)
        Returns:
            Tuple of (x, y) coordinates
        """
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Epsilon-greedy
        if use_epsilon and random.random() < self.epsilon:
            move_idx = random.choice(valid_moves)
        else:
            # Get Q-values from network
            state_tensor = self.board_to_tensor(self.board)
            with torch.no_grad():
                q_values = self.network(state_tensor.unsqueeze(0))
            
            # Mask invalid moves with very low Q-values
            q_array = q_values[0].cpu().numpy()
            for i in range(self.board_size * self.board_size):
                if i not in valid_moves:
                    q_array[i] = -1e9
            
            move_idx = np.argmax(q_array)
        
        x = move_idx // self.board_size
        y = move_idx % self.board_size
        return (x, y)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.cat([item[0].unsqueeze(0) for item in batch]).to(self.device)
        actions = torch.tensor([item[1] for item in batch], device=self.device)
        rewards = torch.tensor([item[2] for item in batch], device=self.device, dtype=torch.float32)
        next_states = torch.cat([item[3].unsqueeze(0) for item in batch]).to(self.device)
        dones = torch.tensor([item[4] for item in batch], device=self.device, dtype=torch.float32)
        
        # Current Q-values
        q_values = self.network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Calculate loss and backpropagate
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.total_loss += loss.item()
        self.update_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.network.state_dict())
    
    def save_model(self, filepath: str):
        """Save model weights to file"""
        torch.save(self.network.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights from file"""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.network.state_dict())
        print(f"Model loaded from {filepath}")
    
    def get_average_loss(self):
        """Get average loss for the current batch"""
        if self.update_count == 0:
            return 0
        avg_loss = self.total_loss / self.update_count
        self.total_loss = 0
        self.update_count = 0
        return avg_loss
