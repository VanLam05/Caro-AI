# Gomoku AI Game - Hướng Dùng

## Tính Năng Chính

### 1. **Chế Độ Chơi** (Lúc Bắt Đầu)
- **PvP (Người vs Người)**: 2 người chơi cùng nhau
- **PvAI (Người vs AI)**: Chơi với AI

### 2. **Chế Độ AI** (Khi chọn PvAI)
**Độ Khó (Difficulty)**:
- **EASY** (Dễ): AI chọn nước đi ngẫu nhiên
- **MEDIUM** (Trung Bình): AI dùng heuristic evaluation
- **HARD** (Khó): AI dùng minimax/minimax với alpha-beta pruning

**Ai Đi Trước**:
- **Human First** (Mặc định): Người đi trước, AI đi thứ 2 → Nước của người là X, AI là O
- **AI First**: AI đi trước, người đi thứ 2 → Nước của AI là X, người là O

### 3. **Các Nút Chức Năng** (Trong Lúc Chơi)
- **Undo**: Hoàn tác nước đi
  - **PvP**: Hoàn tác 1 nước cuối cùng
  - **PvAI**: Hoàn tác 2 nước (nước người + nước AI) để quay lại lượt của người
- **Exit**: Thoát game
- **Replay**: Chơi lại từ đầu (quay về menu)

### 4. **Luật Chơi**
- Bàn cơ tả 15x15 ô
- Cần 5 quân liên tiếp (ngang, dọc, chéo) để thắng
- Người có quân X đi trước (mặc định)

## Cách Chơi
1. **Khởi động game** → Chọn PvP hoặc PvAI
2. **Nếu chọn PvAI**:
   - Chọn độ khó (easy/medium/hard) - mặc định easy
   - Chọn ai đi trước (human/AI) - mặc định human
3. **Chơi game**:
   - Click chuột trên ô cần đặt quân
   - Khi game kết thúc, nút Replay/Exit sẽ hiển thị

## Ghi Chú Kỹ Thuật
- Bấm `Alt+F4` hoặc click X để thoát
- Nước đi cuối cùng được highlight màu xanh (GREEN)
- Các nút được vô hiệu hóa khi không thể dùng
- AI sẽ có 500ms delay khi suy nghĩ (để dễ quan sát)

## TODO (Cần Hoàn Thiện)
- [ ] Cải thiện độ khó AI (MEDIUM/HARD còn cần được implement)
- [ ] Thêm timer/score tracking
- [ ] Optimization cho AI moves
- [ ] Network support cho remote play
