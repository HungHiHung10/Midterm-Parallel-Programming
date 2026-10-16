# Nhóm 22 – phần việc thành viên 3

Bản thực hiện gồm khung chương trình, log màu, dữ liệu chuẩn hóa, kiểm thử tích hợp, profiling và notebook có output thật. `Group22_Midterm.ipynb` là notebook báo cáo duy nhất của nhóm.

## Chạy tại máy

```powershell
python -m pip install -r requirements.txt
python tools/prepare_dataset.py
python tools/build.py
python tools/verify_cli.py
python tools/execute_notebook.py
python tools/package.py
```

Nếu đã có đủ prepared/synthetic thì không cần chuẩn bị lại. Bộ đọc HEIC cục bộ tại `.tools/` không đưa vào ZIP; cài bằng requirements khi chuyển máy. `prepare_dataset.py` giữ nguyên ảnh raw và không ghi đè synthetic đã có. Nếu chỉ có một phần synthetic, script dừng để tránh phá dữ liệu.

CLI trên Windows:

```powershell
.\stitch.exe dataset/prepared/medium/site1_indoor_left.png dataset/prepared/medium/site1_indoor_right.png outputs_v0/panorama_site1_indoor.png --version 0 --runs 5 --warmup 1 --save-h outputs_v0/H.txt --timings-csv outputs_v0/timing.csv
```

Linux/Colab dùng `./stitch`. H luôn ánh xạ **right B → left A**. Mặc định warmup = 0; công cụ benchmark/notebook đặt warmup = 1 và runs = 5. Thư mục đầu ra phải tồn tại. CLI trả mã lỗi khi không ghép được hoặc không ghi được ảnh/H/CSV. `NO_COLOR=1` tắt ANSI; mặc định SUCCESS xanh, ERROR đỏ, WARN vàng, PROCESS trắng.

V0 có thể biên dịch không cần GPU/CUDA:

```bash
g++ -O3 -std=c++14 -x c++ src/main.cu src/stb_impl.cpp src/v0_frontend.cpp src/v0_backend.cpp src/v0_stitching.cpp -o stitch
# Hoặc: cd src && make cpu
# Có CUDA: cd src && make cuda
```

## Dữ liệu và kết quả

- 24 ảnh PNG: `dataset/prepared/{small,medium,large}/`; manifest ghi nguồn HEIC, SHA256, crop box và kiểm tra upscale.
- `dataset/synthetic/`: fixture tự sinh với H dịch chuyển 320 px, seed cố định; provenance ghi rõ không phải tài nguyên môn học.
- `outputs_v0/`: bốn panorama medium, CSV từng lần chạy, bảng tổng hợp, H, log, JSON ghi phần cứng/compiler/hash nguồn và biểu đồ.
- `team.json`: thông tin nhóm, vai trò, địa điểm và URL dataset. TV1: Trần Nguyễn Minh Quân; TV2: Vũ Trần Phúc; TV3: Nguyễn Hữu Khánh Hưng.
- `dist/Group22_Midterm.zip`: notebook có output, src, tools và tài liệu hỗ trợ. `dist/Group22_Dataset.zip`: raw + prepared + synthetic. Dataset nằm ngoài ZIP mã nguồn theo yêu cầu link dataset trong đề.

## Colab

Giải nén hai ZIP vào cùng `/content/Group22_Midterm`, mở notebook và Run all. Hoặc đặt các thư mục/file tương ứng vào `MyDrive/Group22_Midterm`; sửa `DRIVE_PROJECT` trong setup cell rồi Run all để mount và copy. Notebook biên dịch trực tiếp từ `src/`, không sinh bản sao thuật toán C++ trong cell. Cần dataset hiện diện, không chỉ notebook/src.

Đã chạy notebook bằng kernel Python trên Windows; chưa có phiên Colab để xác nhận trực tiếp. Đo lại trên Colab sẽ thay đổi số liệu theo CPU/compiler. Phần CUDA utilities chưa được chạy thử với nvcc/GPU tại máy này.

## Các việc còn phụ thuộc nhóm

1. Cung cấp synthetic pair chính thức từ môn học nếu được yêu cầu. Giữ đúng chiều H B→A; nếu GT là A→B thì nghịch đảo trước khi so sánh, cập nhật provenance và chạy lại notebook. Fixture dịch chuyển hiện tại chỉ kiểm tra tích hợp, không chứng minh trường hợp projective tổng quát.
2. Upload `Group22_Dataset.zip` vào [Drive nhóm](https://drive.google.com/drive/folders/1tAZe8d2SR47yGxrKt7KvgJqyN25M6P1X?usp=sharing), kiểm tra quyền Anyone with the link. Chưa upload hay xác minh quyền truy cập từ môi trường này.
3. Xác nhận phân loại `site2_outdoor`: ảnh tầng 2 tòa E hướng vào hành lang; `site4_challenging` hướng ra cây xanh. Chưa có chứng cứ việc đăng ký địa điểm với lớp. Không tự đổi nhãn thư mục hoặc khẳng định đã đạt tiêu chí đăng ký.
4. TV1/TV2 duyệt các giới hạn V0: backend chưa refit H theo toàn bộ inlier, đang average 50/50 thay cho feathering; frontend chưa bất biến xoay/tỷ lệ, xử lý descriptor biên bằng zero. Ảnh đầu ra có vệt nối/bóng kép và notebook ghi rõ điều này.
