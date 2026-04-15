# 🎮 Gomoku AI - Reinforcement Learning Training

## 📋 Giới Thiệu

Hệ thống huấn luyện AI chơi cờ caro sử dụng **Deep Q-Learning**. AI học bằng cách:
1. Chơi với MiniMax Agent (opponent)
2. Ghi nhớ kinh nghiệm (experience replay)
3. Cải thiện qua neural network training
4. Lưu model weights để dùng lại

---

## 🎯 Kết Quả

✅ **System Ready**: Tất cả components hoạt động  
✅ **Model Trainable**: Neural network có 247,265 parameters  
✅ **Fast Training**: ~5-10 phút cho 100 episodes trên Mac CPU  
✅ **Persistent Storage**: Model weights lưu vào `models/rl_model.pth`  

---

## 🚀 Bắt Đầu Nhanh

### Chạy Training

```bash
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI
bash train.sh
```

Hoặc:
```bash
python3.12 -m pipeline.train
```

### Đánh Giá Model

```bash
bash eval.sh
```

Hoặc:
```bash
python3.12 -m pipeline.inference
```

---

## 📁 Cấu Trúc Dự Án

```
GomokuAI/
├── models/
│   ├── agentRL.py          # Deep Q-Network + RL Agent ⭐
│   ├── agentMiniMax.py     # MiniMax opponent
│   └── rl_model.pth        # Model weights (created after training)
│
├── pipeline/
│   ├── train.py            # Training script ⭐
│   ├── inference.py        # Evaluation script ⭐
│   ├── utils.py            # Helper functions
│   └── __init__.py
│
├── game/
│   ├── board.py
│   ├── buttons.py
│   ├── main.py
│   └── __init__.py
│
├── train.sh                # Shell script để train ⭐
├── eval.sh                 # Shell script để evaluate ⭐
├── test_setup.py           # Verify system setup
├── QUICK_START.md          # Quick start guide ⭐
├── README_TRAINING.md      # Detailed guide
├── INSTALL.md              # Installation guide
└── requirements.txt        # Dependencies
```

---

## 📊 Architecture

### Neural Network

```
Input (225) → FC(256) → ReLU
           ↓
       FC(256) → ReLU
           ↓
       FC(256) → ReLU
           ↓
       Output (225) [Q-values]
```

**15x15 board = 225 cells = 225 possible actions**

### Training Loop

```
1. Initialize: agentRL + agentMiniMax
2. For each episode:
   a. Play game (self-play)
   b. Record experiences (state, action, reward, next_state)
   c. Train on batch from replay memory
   d. Update target network (every N episodes)
3. Save model weights
```

---

## 💾 Model Management

### Save Model (Automatic)
```python
# Lấy sau training
rl_agent.save_model("models/rl_model.pth")
```

### Load Model (Manual)
```python
from models.agentRL import AgentRL
agent = AgentRL(board)
agent.load_model("models/rl_model.pth")
agent.epsilon = 0  # Disable exploration

# Get best move
move = agent.select_move(player=1, use_epsilon=False)
```

---

## 🔧 Hyperparameters

| Tham Số | Giá Trị | Tác Dụng |
|---------|--------|---------|
| NUM_EPISODES | 100 | Số trận chơi (↑ = train lâu hơn) |
| learning_rate | 0.001 | Tốc độ học (↓ = ổn định hơn) |
| gamma | 0.99 | Discount factor (0.9-0.99) |
| epsilon_decay | 0.995 | Exploration giảm (↓ = giảm nhanh) |
| hidden_dim | 256 | Neurons ở hidden layer (↑ = model lớn) |
| batch_size | 64 | Training batch size |
| MiniMax depth | 3 | Opponent strength (↑ = mạnh hơn) |

### Tùy Chỉnh

Mở `pipeline/train.py`:
```python
if __name__ == "__main__":
    NUM_EPISODES = 100     # Thay đổi ở đây
    MODEL_PATH = "models/rl_model.pth"
    trainer = Trainer(num_episodes=NUM_EPISODES, model_save_path=MODEL_PATH)
    trainer.train(target_network_update_freq=5)
```

---

## 📈 Training Progress

Trong quá trình training, bạn sẽ thấy:

```
Episode 10/100
  RL Wins: 3, MiniMax Wins: 6, Draws: 1      ← Win rate
  Win Rate (RL): 30.00%                       ← Percentage
  Avg Loss: 0.123456                          ← Network loss
  Epsilon: 0.9950                             ← Exploration rate
  Memory Size: 450                            ← Experience buffer
```

**Mục tiêu**: RL Wins ↑, Avg Loss ↓

---

## 🎮 Sử Dụng Model Trong Game

```python
from game.board import Board
from models.agentRL import AgentRL

# Setup
board = Board(rows=15, cols=15)
rl_agent = AgentRL(board)
rl_agent.load_model("models/rl_model.pth")
rl_agent.epsilon = 0

# Play move
move = rl_agent.select_move(player=1, use_epsilon=False)
board.make_move(move[0], move[1])
```

---

## ❓ FAQ

**Q: Training mất bao lâu?**  
A: 5-10 phút cho 100 episodes trên Mac CPU. Tăng NUM_EPISODES để train lâu hơn.

**Q: Model lưu ở đâu?**  
A: `models/rl_model.pth` (tự động sau training)

**Q: Có thể train thêm không?**  
A: Có! Chạy `python3.12 -m pipeline.train` lần nữa, model sẽ tiếp tục học.

**Q: Làm sao biết model tốt không?**  
A: Chạy `python3.12 -m pipeline.inference`, xem win rate vs MiniMax.

**Q: MiniMax score bao nhiêu?**  
A: Thường ~50-60% win rate là tốt, phụ thuộc MiniMax depth.

**Q: Có thể dùng GPU không?**  
A: Có, cài CUDA version của PyTorch nếu có NVIDIA GPU.

---

## 🐛 Xử Lý Lỗi

| Lỗi | Giải Pháp |
|-----|----------|
| "No module named torch" | Chạy: `python3.12 -m pip install torch` |
| "Model not found" | Huấn luyện trước: `python3.12 -m pipeline.train` |
| "Slow training" | Giảm MiniMax depth hoặc NUM_EPISODES |
| Import error | Chạy từ GomokuAI folder, dùng python3.12 |

---

## 📚 Tham Khảo

- **Deep Q-Learning**: [Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- **PyTorch**: [pytorch.org](https://pytorch.org)
- **MiniMax**: [Wikipedia](https://en.wikipedia.org/wiki/Minimax)

---

## 🎯 Tiếp Theo

1. ✅ **Huấn luyện**: `bash train.sh`
2. ✅ **Đánh giá**: `bash eval.sh`
3. ⏳ **Tùy chỉnh**: Chỉnh NUM_EPISODES, MiniMax depth
4. ⏳ **Nâng cấp**: Thêm CNN, Policy Gradient, Self-play RL vs RL

---

## 📝 Ghi Chú

- Code dùng **Deep Q-Learning** (DQN)
- Experience Replay + Target Network để ổn định training
- Model lưu dạng PyTorch state_dict (.pth)
- Tương thích Python 3.12+

---

**Happy Training! 🚀🎮**
