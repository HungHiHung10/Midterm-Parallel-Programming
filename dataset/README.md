# Dataset Specifications (CSC14120 - Midterm Project)

Tài liệu hướng dẫn chuẩn bị bộ dữ liệu thực nghiệm theo quy định của môn học:

## 1. Yêu cầu bộ dữ liệu (24 ảnh chuẩn tỷ lệ 16:9):
- **4 địa điểm chụp (Registered Sites):**
  - **Site 1 (Indoor):** Chụp trong nhà (phòng học, sảnh tòa nhà, thư viện).
  - **Site 2 (Outdoor):** Chụp ngoài trời (sân trường, công viên, tượng đài).
  - **Site 3 (Custom):** Địa điểm tự chọn với nhiều góc cạnh, chi tiết phong phú.
  - **Site 4 (Challenging):** Địa điểm thử thách (ánh sáng yếu ban đêm, hoa văn lặp lại, hoặc ít chi tiết).
- **Quy cách chụp:**
  - Đứng cố định tại một vị trí và xoay góc máy (Rotate, don't walk) để triệt tiêu hiệu ứng thị sai (Parallax error).
  - Độ chồng lấp (Overlap) giữa hai ảnh liên tiếp từ 30% đến 50%.
- **3 độ phân giải chuẩn (tỷ lệ 16:9):**
  - **Small:** $1280 \times 720$ (4 cặp = 8 ảnh) đặt tại `prepared/small/`
  - **Medium:** $1920 \times 1080$ (4 cặp = 8 ảnh) đặt tại `prepared/medium/`
  - **Large:** $3840 \times 2160$ (4 cặp = 8 ảnh) đặt tại `prepared/large/`
  - Tổng cộng: **24 ảnh chuẩn**.

## 2. Dữ liệu kiểm thử độ chính xác (Synthetic Pair):
- Đặt tại `synthetic/` gồm:
  - `synthetic_left.png`
  - `synthetic_right.png`
  - `H_ground_truth.txt` (ma trận homography $3 \times 3$ chuẩn)

## 3. Lưu trữ & Chia sẻ Google Drive:
- `raw/` dùng cấu trúc phẳng: `indoor_left/right`, `outdoor_left/right`, `custom_left/right`, `challenging_left/right`; phần mở rộng có thể là JPG, PNG hoặc HEIC.
- `backup/` chỉ lưu bản cũ local để khôi phục, bị loại khỏi Git và `Group22_Dataset.zip`.
- Toàn bộ ảnh gốc chất lượng cao (`raw/`) và 24 ảnh chuẩn hóa (`prepared/`) được nén và tải lên Google Drive của nhóm với quyền truy cập "Anyone with the link can view".
- Đường dẫn Google Drive này bắt buộc phải được chèn vào Notebook ở mục 2 và mục 3.6.
