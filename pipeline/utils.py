"""Utility functions for RL training and inference"""

import os
import json
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Log training statistics"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.json"
        self.stats = []
    
    def log_episode(self, episode: int, rl_wins: int, mm_wins: int, draws: int, 
                    avg_loss: float, epsilon: float, memory_size: int):
        """Log episode statistics"""
        stat = {
            'episode': episode,
            'rl_wins': rl_wins,
            'mm_wins': mm_wins,
            'draws': draws,
            'rl_win_rate': 100 * rl_wins / (rl_wins + mm_wins + draws) if (rl_wins + mm_wins + draws) > 0 else 0,
            'avg_loss': avg_loss,
            'epsilon': epsilon,
            'memory_size': memory_size,
            'timestamp': datetime.now().isoformat()
        }
        self.stats.append(stat)
        self.save()
    
    def save(self):
        """Save logs to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


class ModelManager:
    """Manage model saving and loading"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def get_latest_model(self) -> str:
        """Get path to latest saved model"""
        models = list(self.model_dir.glob("rl_model_*.pth"))
        if models:
            return str(max(models, key=os.path.getctime))
        return None
    
    def get_model_path(self, name: str = "rl_model.pth") -> str:
        """Get model path"""
        return str(self.model_dir / name)
    
    def list_models(self):
        """List all saved models"""
        models = list(self.model_dir.glob("*.pth"))
        return [str(m) for m in sorted(models)]


def print_board(grid, rows: int = 15, cols: int = 15):
    """Print board for visualization"""
    print("\n  ", end="")
    for j in range(cols):
        print(f"{j:2d}", end=" ")
    print()
    
    for i in range(rows):
        print(f"{i:2d} ", end="")
        for j in range(cols):
            cell = grid[i][j]
            if cell == 1:
                print(" X", end=" ")
            elif cell == -1:
                print(" O", end=" ")
            else:
                print(" .", end=" ")
        print()
    print()


if __name__ == "__main__":
    # Test utilities
    logger = TrainingLogger()
    logger.log_episode(1, 5, 3, 2, 0.123, 0.99, 500)
    
    manager = ModelManager()
    print(f"Model will be saved to: {manager.get_model_path()}")
