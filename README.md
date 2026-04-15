# Gomoku AI - Reinforcement Learning Training System

🎮 **Huấn luyện AI chơi cờ caro bằng Deep Q-Learning**

> Tạo AI tự học để chơi cờ caro bằng self-play với MiniMax Agent + neural network training.

---

## 📘 Hướng Dẫn Bắt Đầu

### 1️⃣ **MỚI LẦN ĐẦU?** → Đọc [QUICK_START.md](QUICK_START.md)

5 phút để hiểu cơ bản và chạy training.

### 2️⃣ **MUỐN CHI TIẾT?** → Đọc [README_TRAINING.md](README_TRAINING.md)

Hyperparameters, architecture, recommendations.

### 3️⃣ **CẦN CÀI ĐẶT?** → Đọc [INSTALL.md](INSTALL.md)

Cài PyTorch và dependencies.

### 4️⃣ **TỔNG QUAN?** → Đọc [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

Architecture, folder structure, usage.

---

## 🚀 Chạy Ngay (1 Tập Lệnh)

```bash
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI

# Huấn luyện AI
bash train.sh

# Đánh giá Model
bash eval.sh
```

---

## 📦 Cấu Trúc Files

| File | Mô Tả |
|------|-------|
| **agentRL.py** | Deep Q-Network + RL Agent |
| **train.py** | Training script chính |
| **inference.py** | Đánh giá model |
| **rl_model.pth** | Model weights (tạo sau training) |
| **train.sh / eval.sh** | Shell scripts |
| **test_setup.py** | Verify system |

---

## 🎯 Chứng Năng Chính

✅ Deep Q-Learning training  
✅ Self-play với MiniMax opponent  
✅ Experience replay + Target network  
✅ Model persistence (save/load weights)  
✅ Training evaluation/stats  

---

## 💻 Yêu Cầu

- Python 3.12+ (cần 3.12 vì PyTorch cài cho 3.12)
- PyTorch 2.0+
- NumPy
- Mac/Linux/Windows

---

## 🔄 Workflow

```
1. Setup Environment
   └─ pip install torch numpy

2. Train Model
   └─ python3.12 -m pipeline.train
   └─ Weights saved: models/rl_model.pth

3. Evaluate Model
   └─ python3.12 -m pipeline.inference
   └─ Check win rate vs MiniMax

4. Use in Game
   └─ Load model
   └─ Get best moves
   └─ Integrate to your game
```

---

## 📊 Điều Gì Sẽ Xảy Ra

### Training
- AI RL chơi 100 trận với MiniMax
- Ghi nhớ game trace → experience
- Train neural network 10x per episode
- Lưu best weights → models/rl_model.pth
- **Kết quả**: Win rate 30-60% vs MiniMax

### Inference
- Load model từ file
- Chơi 5 trận test
- In ra statistics

---

## ⚡ Quick Commands

```bash
# Kiểm tra setup
python3.12 test_setup.py

# Train
python3.12 -m pipeline.train

# Evaluate
python3.12 -m pipeline.inference

# Hoặc dùng shell scripts
bash train.sh
bash eval.sh
```

---

## 🎓 Cách Hoạt Động

### Deep Q-Learning

```
1. State: 15x15 board = 225 cells
2. Action: Place piece at any empty cell = 225 actions
3. Network: 4-layer NN (225 → 256 → 256 → 256 → 225)
4. Training: Minimize loss between predicted Q và target Q
5. Epsilon-greedy: Balance exploration vs exploitation
```

### Self-Play Training

```
RL Agent vs MiniMax Agent
    ↓
Play Game
    ↓
Collect Experiences
    ↓
Experience Replay (sample batches)
    ↓
Update Network with MSE Loss
    ↓
Repeat!
```

---

## 📈 Expected Results

**After 100 episodes:**
- RL Win Rate: 30-50% vs MiniMax (depth=3)
- Average Loss: 0.01-0.1 (decreasing)
- Training Time: 5-10 minutes

**Cách cải thiện:**
- Tăng NUM_EPISODES → 500 (train lâu hơn)
- Tăng MiniMax depth → 4 (opponent mạnh hơn)
- Tùy chỉnh learning rate

---

## 🔧 Customization

Mở `pipeline/train.py`:

```python
# Tăng số episodes
NUM_EPISODES = 500  # default: 100

# Tăng MiniMax strength
self.minimax_agent = AgentMiniMax(self.board, max_depth=4)  # default: 3

# Thay model path
MODEL_PATH = "models/custom_model.pth"
```

---

## 💾 Model Usage in Game

```python
from models.agentRL import AgentRL
from game.board import Board

# Load trained model
board = Board()
agent = AgentRL(board)
agent.load_model("models/rl_model.pth")
agent.epsilon = 0  # Disable exploration

# Get AI move
best_move = agent.select_move(player=1, use_epsilon=False)
x, y = best_move
board.make_move(x, y)
```

---

## 📚 Documentation

| Tài Liệu | Nội Dung |
|---------|---------|
| [QUICK_START.md](QUICK_START.md) | 5 phút để chạy training |
| [README_TRAINING.md](README_TRAINING.md) | Chi tiết hyperparameters & architecture |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Tổng quan dự án |
| [INSTALL.md](INSTALL.md) | Cài đặt dependencies |

---

## ❓ FAQ

**Làm sao để train nhanh hơn?**  
→ Giảm MiniMax depth hoặc NUM_EPISODES

**Model ở đâu?**  
→ `models/rl_model.pth` (tự động sau train)

**Có thể train thêm không?**  
→ Có! Run training lần nữa, model tiếp tục học

**Win rate là bao nhiêu?**  
→ 30-60% vs MiniMax (depth=3) là tốt

---

## 🐛 Troubleshooting

```bash
# PyTorch import error
python3.12 -m pip install torch

# Module not found
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI
python3.12 -m pipeline.train

# Slow training
# → Giảm MiniMax depth: 3 → 2
# → Giảm NUM_EPISODES: 100 → 50
```

---

## 📝 Notes

- Dùng **python3.12** (PyTorch cài cho 3.12)
- Model weights ~1-2 MB
- Training memory: ~500-1000 experiences
- CPU training: OK, GPU nhanh hơn

---

## 🎯 Next Steps

1. **Đọc QUICK_START.md** - 5 phút overview
2. **Chạy `bash train.sh`** - Start training
3. **Chạy `bash eval.sh`** - Check results  
4. **Tùy chỉnh hyperparameters** - Improve perf
5. **Integrate vào game** - Use dalam ứng dụng

---

## 📞 Support

- Lỗi cài đặt? → [INSTALL.md](INSTALL.md)
- Muốn custom? → [README_TRAINING.md](README_TRAINING.md)
- Overview? → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**Ready to train? 🚀**

```bash
bash train.sh
```

Happy Training! 🎮✨
