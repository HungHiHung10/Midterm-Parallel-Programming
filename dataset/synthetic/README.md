# Synthetic fixture hiện tại

`synthetic_left.png` và `synthetic_right.png` là hai crop 640×480 của một cảnh hình học tự sinh (seed 14120), dịch ngang 320 px, chồng lấp 50%. Không tải từ tài nguyên môn học; không phải cặp synthetic chính thức. Dùng để kiểm tra tích hợp đủ sáu stage và chiều ma trận.

`H_ground_truth.txt` là ma trận 3×3 row-major **B (right) → A (left)**:

```text
1 0 320
0 1 0
0 0 1
```

`provenance.json` ghi nguồn, seed và chiều H. Sai số được tính sau khi chuẩn hóa H[2,2]=1 bằng Frobenius và sai số chiếu trung bình/tối đa trên lưới điểm. Threshold mean 1 px là tiêu chí kiểm thử nội bộ.

Khi nhận được cặp chính thức, thay cả hai ảnh và H, sửa provenance (gồm nguồn/URL nếu có), đảm bảo chiều H B→A. Không dùng kết quả dịch chuyển để kết luận pipeline đúng cho mọi homography. Chạy lại notebook để tạo kết quả thật của dữ liệu mới.
