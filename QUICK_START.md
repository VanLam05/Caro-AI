# 🎮 Hướng Dẫn Nhanh - Training AI Cờ Caro

## 📋 Tóm Tắt

Hệ thống này sử dụng **Deep Q-Learning** để huấn luyện AI chơi cờ caro. Quá trình:
- AI RL tự chơi với AI MiniMax
- Ghi nhớ các trận đấu → train neural network
- Lưu weights model vào file → dùng lại lần sau

---

## 🚀 Chạy Training (5 phút)

### 1. Mở Terminal và vào thư mục dự án

```bash
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI
```

### 2. Chạy training (dùng Python 3.12)

```bash
python3.12 -m pipeline.train
```

**Bạn sẽ thấy:**
```
Starting training for 100 episodes...
Model will be saved to: models/rl_model.pth
================================================================================
Episode 10/100
  RL Wins: 3, MiniMax Wins: 6, Draws: 1
  Win Rate (RL): 30.00%
  Avg Loss: 0.123456
  Epsilon: 0.9950
  Memory Size: 450
================================================================================
...
```

### 3. Sau khi training xong

Model được lưu tại: `models/rl_model.pth`

---

## 📊 Kiểm Tra Chất Lượng Model

```bash
python3.12 -m pipeline.inference
```

Lệnh này sẽ:
- Load model từ `models/rl_model.pth`
- Chơi 5 trận với MiniMax
- In ra win rate

**Output:**
```
Playing 5 games for evaluation...
================================================================================

Game 1/5
RL Agent plays as: Player X (1)
RL Agent (1) plays at (7, 7)
MiniMax Agent (-1) plays at (7, 8)
...
✓ RL Agent wins!
Total moves: 45

================================================================================
EVALUATION RESULTS
================================================================================
Total Games: 5
RL Wins: 3 (60.00%)
MiniMax Wins: 2 (40.00%)
Draws: 0
================================================================================
```

---

## 🔧 Tùy Chỉnh Training

### Muốn train lâu hơn?

Mở `pipeline/train.py`, tìm dòng:
```python
NUM_EPISODES = 100  # Tăng lên 500 để train tốt hơn
```

Thay đổi:
```python
NUM_EPISODES = 500  # Train 500 trận thay vì 100
```

### Muốn MiniMax mạnh hơn?

Mở `pipeline/train.py`, tìm dòng:
```python
self.minimax_agent = AgentMiniMax(self.board, max_depth=3)
```

Thay đổi:
```python
self.minimax_agent = AgentMiniMax(self.board, max_depth=4)  # Mạnh hơn nhưng chậm hơn
```

---

## 📁 Các File Quan Trọng

| File | Mục Đích |
|------|---------|
| `models/agentRL.py` | Deep Q-Network + RL training |
| `pipeline/train.py` | Script huấn luyện chính |
| `pipeline/inference.py` | Đánh giá model |
| `models/rl_model.pth` | Model weights (tạo sau training) |

---

## 💡 Những Điều Cần Biết

✅ **Lần đầu**: Sẽ mất 5-10 phút tùy máy  
✅ **Model được lưu lại**: Có thể load và dùng sau  
✅ **Có thể train thêm**: Model sẽ tiếp tục học từ checkpoint  
✅ **CPU là được**: PyTorch tự động tối ưu trên Mac  

---

## ❓ Troubleshooting

### "No module named 'pipeline'"
```bash
# Chắc chắn bạn đang ở thư mục GomokuAI
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI
python3.12 -m pipeline.train
```

### Training quá chậm?
- Giảm MiniMax depth từ 3 → 2
- Giảm NUM_EPISODES từ 100 → 50
- Dùng Python3.12 (tốt hơn 3.13)

### Model không tìm thấy
```bash
# Chạy training trước
python3.12 -m pipeline.train

# Sau đó mới chạy inference
python3.12 -m pipeline.inference
```

---

## 📖 Thêm Thông Tin

- Chi tiết hơn: xem [README_TRAINING.md](README_TRAINING.md)
- Cài đặt: xem [INSTALL.md](INSTALL.md)

---

## 🎯 Bước Tiếp Theo

1. **Huấn luyện**: `python3.12 -m pipeline.train`
2. **Đánh giá**: `python3.12 -m pipeline.inference`
3. **Tùy chỉnh**: Chỉnh sửa NUM_EPISODES hoặc max_depth
4. **Sử dụng model**: Load vào game của bạn

Chúc bạn thành công! 🎮✨

