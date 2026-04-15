# 🚀 START HERE

## ✅ Mọi Thứ Đã Sẵn Sàng!

Hệ thống Reinforcement Learning để huấn luyện AI cờ caro đã hoàn thành.

---

## ⚡ 30 Giây Để Bắt Đầu

```bash
# Vào thư mục
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI

# Huấn luyện AI
bash train.sh

# Cài có lệnh trên lỗi? → Xem dưới "Troubleshoot"
```

---

## 📊 Khi Training Hoàn Thành

```bash
# Kiểm tra chất lượng model
bash eval.sh
```

Bạn sẽ thấy:
```
RL Wins: X (ZZ%)
MiniMax Wins: Y (WW%)
```

---

## 📚 Tài Liệu

| Nếu Bạn | Đọc File |
|---------|----------|
| 🆕 Chưa biết bắt đầu ở đâu | [QUICK_START.md](QUICK_START.md) |
| 🎓 Muốn hiểu chi tiết | [README_TRAINING.md](README_TRAINING.md) |
| 📋 Muốn tổng quan | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |
| 🔨 Có vấn đề cài đặt | [INSTALL.md](INSTALL.md) |
| 📄 Tất cả chi tiết | [SETUP_COMPLETE.md](SETUP_COMPLETE.md) |

---

## 🐛 Troubleshoot

### "No such file bash train.sh"
```bash
# Chắc chắn bạn ở thư mục đúng:
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI

# Test:
ls train.sh  # Phải thấy file này
bash train.sh  # Chạy
```

### "ModuleNotFoundError: no module named 'torch'"
```bash
python3.12 -m pip install torch
```

### "No module named 'pipeline'"
```bash
# Chắc chắn chạy từ thư mục GomokuAI:
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI

# Kiểm tra:
ls pipeline/train.py  # Phải thấy

# Chạy với python3.12:
python3.12 -m pipeline.train
```

### Training quá chậm?
```bash
# Edit: pipeline/train.py
# Line 138: NUM_EPISODES = 50  (từ 100 → 50)

# Edit: pipeline/train.py  
# Line 24: max_depth=2  (từ 3 → 2)

# Run again:
bash train.sh
```

---

## 📁 Cấu Trúc Trong Một Nôm Na

```
GomokuAI/
├── models/
│   ├── agentRL.py        ⭐ AI học bằng RL
│   ├── agentMiniMax.py   ⭐ Opponent
│   └── rl_model.pth      💾 Model weights (sau khi train)
│
├── pipeline/
│   ├── train.py          🏋️ Script huấn luyện
│   └── inference.py      📊 Script kiểm tra
│
├── train.sh              🚀 Chạy training (đơn giản nhất)
├── eval.sh               📈 Chạy evaluation
│
└── *.md                  📚 Các tài liệu
```

---

## 🎯 3 Bước DUY NHẤT

### Bước 1: Huấn Luyện
```bash
bash train.sh
# Chờ 5-10 phút...
```

### Bước 2: Kiểm Tra
```bash
bash eval.sh
# Xem AI thắng bao nhiêu %
```

### Bước 3: Hoàn Tất!
```
Model được lưu: models/rl_model.pth
Dùng được luôn trong game! 🎮
```

---

## 💻 Hệ Thống Đã Test

✅ Python: 3.12  
✅ PyTorch: 2.11  
✅ NumPy: 2.2  
✅ macOS environment  
✅ 247,265 network parameters  
✅ All imports working  

---

## 🎮 Dùng AI Trong Game

```python
from models.agentRL import AgentRL

agent = AgentRL(board)
agent.load_model("models/rl_model.pth")

# Lấy nước đi tốt nhất
move = agent.select_move(player=1, use_epsilon=False)
```

---

## ❓ Hỏi Gì Tiếp?

- "Training mất bao lâu?" → QUICK_START.md
- "Model lưu ở đâu?" → Tự động: models/rl_model.pth
- "Làm sao cải thiện?" → README_TRAINING.md
- "Có lỗi gì đó" → Xem Troubleshoot ở trên

---

## 🏃 NGAY BÂY GIỜ

```bash
cd /Users/apple/Documents/Kì_2_năm_3/GomokuAI
bash train.sh
```

**That's it!** ✨

Training sẽ bắt đầu. Chờ xong xem kết quả với `bash eval.sh`.

---

**Questions?** 📖
- QUICK_START.md
- README_TRAINING.md  
- PROJECT_SUMMARY.md

**Ready?** 🚀
`bash train.sh`

---

*Happy Training! 🎉*
