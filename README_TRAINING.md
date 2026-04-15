# Huấn Luyện AI Cờ Caro với Reinforcement Learning

## Tổng Quan

Hệ thống này sử dụng Deep Q-Learning để huấn luyện AI chơi cờ caro. Quá trình huấn luyện:
- **Self-play**: AI (RL Agent) chơi với MiniMax Agent
- **Experience Replay**: Lưu trữ các kinh nghiệm và học từ chúng
- **Model Checkpoint**: Lưu weights mô hình sau mỗi lần huấn luyện

## Cài Đặt Requirements

```bash
pip install torch numpy
```

**Hoặc nếu bạn có GPU (CUDA):**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Cấu Trúc Tệp

```
GomokuAI/
├── models/
│   ├── agentRL.py          # Deep Q-Network Agent + Training logic
│   ├── agentMiniMax.py     # MiniMax Agent (opponent)
│   └── rl_model.pth        # Saved model weights (generated)
├── pipeline/
│   ├── train.py            # Script huấn luyện
│   └── inference.py        # Script đánh giá model
├── game/
│   ├── board.py
│   └── main.py
└── README_TRAINING.md      # File này
```

## Huấn Luyện Model

### 1. Chạy training script

```bash
cd /path/to/GomokuAI
python -m pipeline.train
```

**Output example:**
```
Starting training for 100 episodes...
Model will be saved to: models/rl_model.pth
--------------------------------------------------------------------------------
Episode 10/100
  RL Wins: 3, MiniMax Wins: 6, Draws: 1
  Win Rate (RL): 30.00%
  Avg Loss: 0.123456
  Epsilon: 0.9950
  Memory Size: 450
```

### 2. Tùy Chỉnh Tham Số Huấn Luyện

Mở `pipeline/train.py` và chỉnh sửa các tham số:

```python
if __name__ == "__main__":
    NUM_EPISODES = 100          # Số trận chơi (tăng để train lâu hơn, ~500 để tốt)
    MODEL_PATH = "models/rl_model.pth"  # Đường dẫn lưu model
    
    trainer = Trainer(num_episodes=NUM_EPISODES, model_save_path=MODEL_PATH)
    trainer.train(target_network_update_freq=5)
```

## Sử Dụng Model Đã Huấn Luyện

### 1. Đánh Giá Model Trên MiniMax

```bash
python -m pipeline.inference
```

Lệnh này sẽ:
- Load model từ `models/rl_model.pth`
- Chơi 5 trận với MiniMax AI
- In ra kết quả win rate

### 2. Sử Dụng Model Trong Game

```python
from game.board import Board
from models.agentRL import AgentRL

# Tạo board
board = Board(rows=15, cols=15)

# Khởi tạo RL Agent với model đã train
rl_agent = AgentRL(board)
rl_agent.load_model("models/rl_model.pth")
rl_agent.epsilon = 0  # Không dùng exploration

# Lấy nước đi tốt nhất
best_move = rl_agent.select_move(player=1, use_epsilon=False)
print(f"Best move: {best_move}")  # (x, y)
```

## Các Tệp Chính

### 1. `agentRL.py` - Neural Network + RL Training

**Chủ yếu:**
- `DQNNetwork`: Neural network với 4 fully connected layers
- `AgentRL`: Q-Learning agent với experience replay

**Phương thức:**
```python
# Lấy nước đi
move = agent.select_move(player=1, use_epsilon=False)

# Huấn luyện
agent.replay()

# Lưu/Load model
agent.save_model("path/to/model.pth")
agent.load_model("path/to/model.pth")
```

### 2. `train.py` - Huấn Luyện

**Chủ yếu:**
- Tạo self-play games giữa RL agent và MiniMax
- Lưu kinh nghiệm vào replay memory
- Huấn luyện neural network
- Cập nhật target network định kỳ
- Lưu model weights

### 3. `inference.py` - Đánh Giá

**Chủ yếu:**
- Load model đã train
- Chơi games để đánh giá hiệu suất
- In ra statistics

## Hyperparameters

Các tham số quan trọng có thể điều chỉnh trong `agentRL.py`:

```python
# Trong __init__
learning_rate = 0.001          # Learning rate (tăng → học nhanh hơn nhưng không ổn định)
gamma = 0.99                   # Discount factor (0.9-0.99)
epsilon = 1.0                  # Exploration rate
epsilon_decay = 0.995          # Từng bước giảm epsilon
epsilon_min = 0.01             # Giới hạn dưới của epsilon

# Trong DQNNetwork
hidden_dim = 256               # Số neurons ở hidden layers
batch_size = 64                # Batch size để training
```

## Các Vấn Đề Thường Gặp

### 1. "Model not found"
- Đảm bảo bạn đã huấn luyện model bằng `python -m pipeline.train`
- Hoặc chỉ định đúng model path

### 2. "GPU out of memory"
- Giảm `batch_size` hoặc `hidden_dim` trong `agentRL.py`
- Hoặc chạy trên CPU bằng cách comment dòng `cuda` check

### 3. Model không cải thiện
- Tăng `NUM_EPISODES` (train lâu hơn)
- Điều chỉnh learning rate
- Tăng MiniMax depth để tạo dữ liệu huấn luyện tốt hơn

## Tiếp Theo

1. **Tăng cường Data**: Tăng `NUM_EPISODES` từ 100 lên 500-1000
2. **Self-play nâng cao**: Huấn luyện RL vs RL (thay vì vs MiniMax)
3. **Policy Gradient**: Thay thế Q-Learning bằng A3C hoặc PPO
4. **Board Representation**: Sử dụng CNN thay vì fully connected

## Tham Khảo

- Deep Q-Learning: https://en.wikipedia.org/wiki/Q-learning
- PyTorch: https://pytorch.org/
- MiniMax Algorithm: https://en.wikipedia.org/wiki/Minimax
