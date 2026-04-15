# Hướng Dẫn Cài Đặt

## 1. Cài Đặt Dependencies

### Cách 1: Cài Trên CPU (Đơn Giản - Khuyên Dùng)

```bash
# Trên macOS hoặc Linux
pip install -r requirements.txt

# Hoặc cài trực tiếp:
pip install torch numpy
```

### Cách 2: Cài Với GPU CUDA (Nếu Có GPU)

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Cách 3: Cài MPS (Apple Metal Performance Shaders - MacBook M1/M2/M3)

```bash
pip install torch numpy
# PyTorch tự động dùng MPS trên macOS nếu có
```

## 2. Xác Minh Cài Đặt

```bash
# Chạy test setup
python test_setup.py
```

Bạn sẽ thấy output như:
```
============================================================
Testing Gomoku AI Training Setup
============================================================
✓ All imports successful
✓ Board created: 15x15
✓ RL Agent created
  - Device: cpu
  - Network parameters: 331,264
✓ MiniMax Agent created
✓ Board to tensor: shape torch.Size([225])
✓ Network forward pass: output shape torch.Size([1, 225])
✓ Move selection: (7, 7)
✓ Valid moves count: 225

============================================================
✓ All tests passed! Ready to train.
============================================================

To start training, run:
  python -m pipeline.train

To evaluate a trained model, run:
  python -m pipeline.inference
```

## 3. Chạy Training

```bash
# Huấn luyện AI
python -m pipeline.train

# Hoặc chỉ định số episodes
# (Sửa NUM_EPISODES trong pipeline/train.py)
```

## 4. Đánh Giá Model

```bash
# Đánh giá model đó được train
python -m pipeline.inference
```

## 5. Lưu Ý Quan Trọng

- **Training lần đầu sẽ mất vài phút** (tuỳ CPU)
- Model được lưu tại: `models/rl_model.pth`
- Nếu muốn train lâu, tăng `NUM_EPISODES` trong `pipeline/train.py`
- Nếu gặp lỗi memory, giảm `hidden_dim` hoặc `batch_size` trong `agentRL.py`

## Troubleshooting

### PyTorch không cài được

```bash
# Xóa PyTorch cũ
pip uninstall torch

# Cài lại
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "ModuleNotFoundError: No module named 'game'"

Chắc chắn bạn chạy commands từ thư mục GomokuAI:
```bash
cd /path/to/GomokuAI
python test_setup.py
```

### Training quá chậm

- Giảm MiniMax depth từ 3 → 2 trong `train.py`
- Giảm số episodes từ 100 → 50
- Sử dụng GPU (cài CUDA version của PyTorch)
