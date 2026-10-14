# Synthetic Test Dataset (Correctness Evaluation)

Thư mục này dùng để lưu trữ cặp ảnh nhân tạo và ma trận Ground-truth Homography:

1. `synthetic_left.png`: Ảnh nhân tạo bên trái.
2. `synthetic_right.png`: Ảnh nhân tạo bên phải (được biến đổi từ ảnh gốc bằng ma trận $H_{GT}$).
3. `H_ground_truth.txt`: Tệp văn bản chứa ma trận $3 \times 3$ chuẩn (lưu theo dạng ma trận 3 dòng, mỗi dòng 3 số thực phân cách bằng dấu cách):
```
h11 h12 h13
h21 h22 h23
h31 h32 h33
```

Khi chạy chương trình:
```bash
./src/stitch dataset/synthetic/synthetic_left.png dataset/synthetic/synthetic_right.png outputs_v0/synthetic_pano.png --version 0 --save-h outputs_v0/H_estimated.txt
```
Notebook sẽ đọc `H_ground_truth.txt` và `H_estimated.txt` để tính chuẩn sai số Frobenius Norm hoặc điểm chiếu trung bình (Projection Error) theo yêu cầu mục 3.5.
