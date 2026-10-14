# KẾ HOẠCH PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN GIỮA KỲ (MIDTERM ASSIGNMENT)
**Môn học:** CSC14120 – Lập trình song song (Parallel Programming)  
**Đề tài:** Panorama Image Stitching (Ghép ảnh toàn cảnh)  
**Trọng số:** 10% tổng điểm đồ án | **Hạn nộp:** 23:59 Cuối tuần thứ 7  
**Repository:** [https://github.com/HungHiHung10/Midterm-Parallel-Programming.git](https://github.com/HungHiHung10/Midterm-Parallel-Programming.git)

### Thông tin nhóm (Mục 3.1 — Bắt buộc)

| Vai trò | Họ tên | MSSV |
| :--- | :--- | :--- |
| Thành viên 1 (Back-end & Algorithmic Core Lead) | `[Điền tên]` | `[Điền MSSV]` |
| Thành viên 2 (Front-end & Computer Vision Core Lead) | `[Điền tên]` | `[Điền MSSV]` |
| Thành viên 3 (System, Data, Profiling & Notebook Lead) | `[Điền tên]` | `[Điền MSSV]` |

---

## 1. MỤC TIÊU VÀ SẢN PHẨM BÀN GIAO (DELIVERABLES)

Mục tiêu của bài nộp Giữa kỳ là xây dựng thành công **phiên bản tuần tự (V0)** làm mốc chuẩn (baseline), đo lường thời gian thực thi chi tiết của 6 giai đoạn trên 3 độ phân giải khác nhau, kiểm chứng tính đúng đắn trên tập ảnh chuẩn, và lập kế hoạch song song hóa cụ thể trên GPU (CUDA Plan) cho các phiên bản tiếp theo.

### Sản phẩm nộp bài theo quy định:
Một tệp nén duy nhất: **`GroupXX_Midterm.zip`** gồm:
1. **`GroupXX_Midterm.ipynb`**: Báo cáo dạng Jupyter Notebook (Google Colab), đã lưu toàn bộ output của các ô thực thi.
2. **`src/`**: Toàn bộ mã nguồn C/C++ cần thiết để biên dịch chương trình:
   * `common.h`, `cuda_utils.cuh`, `versions.h`, `main.cu`, `stb_impl.cpp`
   * `v0_<name>.cpp` (hoặc các file module tuần tự đi kèm)
   * Header chú thích: Đầu mỗi file phải ghi rõ tên tác giả và các nguồn mã mở được tham khảo/chuyển thể.
3. **Dataset (Google Drive link)**: Đường dẫn mở đặt trong notebook dẫn tới thư mục chứa ảnh thô và 24 ảnh đã crop/resize + synthetic pair.

---

## 2. KIẾN TRÚC PIPELINE GHÉP ẢNH (6 GIAI ĐOẠN)

```
[Ảnh A] ──> [Stage 1: Preprocessing] ──> [Stage 2: Keypoints] ──> [Stage 3: Descriptors] ──┐
                                                                                             │
[Ảnh B] ──> [Stage 1: Preprocessing] ──> [Stage 2: Keypoints] ──> [Stage 3: Descriptors] ──┼──> [Stage 4: Matching] ──> [Stage 5: RANSAC] ──> [Stage 6: Warping & Blending] ──> [Panorama]
```

* **Stage 1 (Preprocessing):** Chuyển ảnh màu RGB sang Grayscale và làm mịn giảm nhiễu (Gaussian Blur).
* **Stage 2 (Keypoint Detection):** Tìm các điểm đặc trưng nổi bật (FAST / Harris Corner) + lọc Non-Maximum Suppression (NMS) + Top-K.
* **Stage 3 (Description):** Trích xuất vector đặc trưng mô tả vùng lân cận (BRIEF / ORB 256-bit nhị phân).
* **Stage 4 (Matching):** So khớp đặc trưng giữa hai ảnh (Brute-Force khoảng cách Hamming + Lowe's ratio test).
* **Stage 5 (Homography Estimation):** Ước lượng ma trận biến đổi hình học $3\times3$ bằng RANSAC 4 điểm (DLT).
* **Stage 6 (Warping & Blending):** Biến đổi ngược (Inverse Warping với Bilinear Interpolation) và hòa trộn biên ảnh (Linear Feathering Blending).

---

## 3. PHÂN CHIA NHIỆM VỤ CHI TIẾT CHO TỪNG THÀNH VIÊN

---

### THÀNH VIÊN 1: BACK-END & ALGORITHMIC CORE LEAD
*(Phụ trách các giai đoạn 4, 5, 6 của Pipeline tuần tự & Kế hoạch GPU tương ứng)*

#### 1. Trách nhiệm kỹ thuật mã nguồn (C/C++ V0):
* **Cài đặt Stage 4 - Feature Matching:**
  * Hiện thực thuật toán Brute-Force Matcher tính khoảng cách Hamming giữa các vector descriptor 256-bit của ảnh A và ảnh B.
  * Tích hợp **Lowe's ratio test** ($d_1 / d_2 < \text{threshold}$, với $d_1, d_2$ lần lượt là khoảng cách nhỏ nhất và nhỏ thứ nhì, gợi ý $0.75 - 0.8$) để loại bỏ các điểm ghép cặp mơ hồ.
  * Tùy chọn kiểm tra chéo (Cross-check consistency) để tăng độ tin cậy.
* **Cài đặt Stage 5 - Homography Estimation with RANSAC:**
  * Thuật toán RANSAC chuẩn: lặp $N$ lần (ví dụ $1000 - 2000$ iterations). Mỗi lần chọn ngẫu nhiên 4 cặp điểm không đồng tuyến.
  * Giải hệ phương trình DLT (Direct Linear Transformation) bằng phép khử Gauss hoặc SVD để tìm ma trận $H$ kích thước $3 \times 3$.
  * Áp dụng biến đổi chiếu $p' = H \cdot p$ để đếm số lượng điểm khớp đúng (inliers) thỏa khoảng cách sai số Euclidean dưới ngưỡng $\tau$ (ví dụ 3.0 pixels).
  * Lưu lại ma trận $H$ tốt nhất có số inliers lớn nhất; thực hiện bước Refinement tính toán lại $H$ từ toàn bộ tập inliers này.
* **Cài đặt Stage 6 - Warping & Blending:**
  * Tính ma trận nghịch đảo $H^{-1}$.
  * Xác định kích thước canvas bao bọc ảnh ghép (bounding box).
  * Hiện thực kỹ thuật **Inverse Warping**: duyệt qua từng pixel của canvas, dùng $H^{-1}$ chiếu ngược về tọa độ thực trên ảnh nguồn để tránh lỗ hổng pixel (holes).
  * Áp dụng thuật toán nội suy song tuyến tính (**Bilinear Interpolation**) để làm mịn chất lượng ảnh tái tạo.
  * Hiện thực thuật toán hòa trộn tuyến tính (**Linear Feathering / Alpha Blending**) tại vùng chồng lấp (overlap) để khử vệt sáng/tối tại đường ranh giới ghép.

#### 2. Trách nhiệm viết Báo cáo & Thiết kế GPU (Midterm Report):
* Soạn thảo chi tiết thuật toán mục **3.2 (Problem Understanding)** và **3.3 (Chosen Methods)** cho các Stages 4, 5, 6. Phần lý giải phải bao gồm cả **chất lượng đầu ra** (output quality) lẫn **tính phù hợp để song song hóa trên GPU** (GPU suitability) — ví dụ: Brute-force matching có độ phức tạp cao nhưng mỗi cặp so khớp hoàn toàn độc lập nên rất phù hợp GPU; Inverse warping mỗi pixel tính toán riêng biệt nên song song hóa tự nhiên.
* Soạn thảo mục **3.7 (GPU Plan)** cho các Stages 4, 5, 6:
  * **V1 Thread Mapping:** Mỗi thread tính khoảng cách so khớp cho 1 descriptor; mỗi warp hoặc block chạy thử nghiệm 1 giả thuyết RANSAC độc lập; mỗi thread tính toán màu cho 1 output pixel trên canvas warping.
  * **Kế hoạch tối ưu V2/V3:** Tận dụng Tiled Shared Memory cho ma trận descriptor; lệnh nội tại đếm bit `__popc()` cho khoảng cách Hamming; sử dụng CUDA Texture Memory cho phép nội suy song tuyến tính phần cứng khi warping; tối ưu Warp-shuffle reduction để đếm inlier trong RANSAC.
* Lường trước các rủi ro kỹ thuật (Warp divergence trong RANSAC, overhead khi warp canvas kích thước lớn).

#### 3. Tiêu chí hoàn thành (Definition of Done - DoD):
* [ ] Code C++ chạy đúng, sạch sẽ, không rò rỉ bộ nhớ, có comment giải thích và trích dẫn tác giả ở đầu file.
* [ ] Hàm ghép ảnh nhận đầu vào là 2 ảnh và xuất ra ảnh panorama ghép chuẩn xác, không bị méo lệch hay lộ vệt cắt nối rõ rệt.
* [ ] Hoàn thành toàn bộ nội dung lý thuyết và thiết kế GPU của Stages 4, 5, 6 trong Notebook.

---

### THÀNH VIÊN 2: FRONT-END & COMPUTER VISION CORE LEAD
*(Phụ trách các giai đoạn 1, 2, 3 của Pipeline tuần tự & Kế hoạch GPU tương ứng)*

#### 1. Trách nhiệm kỹ thuật mã nguồn (C/C++ V0):
* **Cài đặt Stage 1 - Preprocessing:**
  * Chuyển đổi ảnh màu RGB sang Grayscale theo trọng số độ chói tiêu chuẩn: $Y = 0.299R + 0.587G + 0.114B$.
  * Tạo ma trận Gaussian Kernel (ví dụ kích thước $5\times5$, $\sigma = 1.0 - 1.4$).
  * Thực hiện tích chập làm mịn ảnh để khử nhiễu hạt trước khi phát hiện góc. Khuyến khích cài đặt tích chập tách rời 2 chiều (Separable 1D filter: lọc hàng rồi lọc cột) để tối ưu độ phức tạp từ $O(K^2)$ xuống $O(2K)$.
* **Cài đặt Stage 2 - Keypoint Detection:**
  * Hiện thực thuật toán nhận diện góc: Khuyến khích chọn **FAST (Features from Accelerated Segment Test)** vì tốc độ cao và cực kỳ phù hợp để song song hóa GPU, hoặc chọn **Harris Corner Detector**.
  * Cài đặt cơ chế kiểm tra nhanh: với FAST, xét vòng tròn 16 pixel xung quanh, kiểm tra trước 4 điểm la bàn (1, 5, 9, 13) để loại bỏ sớm các pixel phẳng.
  * Hiện thực thuật toán triệt tiêu cực đại cục bộ (**Non-Maximum Suppression - NMS**) trong cửa sổ $3\times3$ hoặc $5\times5$ để tránh các điểm đặc trưng dồn cục.
  * Cơ chế chọn lọc **Top-K keypoints** (ví dụ giữ lại 1000 - 2000 điểm có điểm số góc cao nhất) để kiểm soát khối lượng tính toán ở các bước sau.
* **Cài đặt Stage 3 - Feature Description:**
  * Hiện thực trích xuất vector đặc trưng **BRIEF** (hoặc ORB đơn giản hóa).
  * Định nghĩa mẫu phân bố ngẫu nhiên chuẩn Gaussian (Gaussian sampling pattern) gồm 256 cặp tọa độ tương đối $(p_i, q_i)$ trong vùng lân cận $31\times31$ xung quanh keypoint.
  * Sinh chuỗi nhị phân 256-bit (lưu trong mảng `uint8_t[32]` hoặc `uint32_t[8]`): bit $i = 1$ nếu $I(p_i) < I(q_i)$, ngược lại bằng $0$.

#### 2. Trách nhiệm viết Báo cáo & Thiết kế GPU (Midterm Report):
* Soạn thảo chi tiết thuật toán mục **3.2 (Problem Understanding)** và **3.3 (Chosen Methods)** cho các Stages 1, 2, 3. Phần lý giải phải bao gồm cả **chất lượng đầu ra** (output quality) lẫn **tính phù hợp để song song hóa trên GPU** (GPU suitability) — ví dụ: BRIEF descriptor 256-bit nhị phân cho phép dùng lệnh `__popc` đếm bit cực nhanh trên GPU; FAST detector chỉ xét vùng lân cận cục bộ nên mỗi pixel xử lý hoàn toàn độc lập.
* Soạn thảo mục **3.7 (GPU Plan)** cho các Stages 1, 2, 3:
  * **V1 Thread Mapping:** Mỗi thread xử lý 1 pixel cho phép chuyển xám và làm mịn; mỗi thread tính điểm phản hồi góc cho 1 pixel; mỗi thread sinh descriptor cho 1 keypoint.
  * **Kế hoạch tối ưu V2/V3:** Stencil 2D Gaussian blur sử dụng Tiled Shared Memory có vùng biên (halos); kỹ thuật Prefix-Sum (Scan) / Stream Compaction để lọc tập điểm đặc trưng thay vì dùng biến đếm nguyên tử toàn cục (`atomicAdd`); sử dụng CUDA Streams để chạy song song Stages 1–3 cho ảnh A và ảnh B cùng một lúc.
* Phân tích các rủi ro: Chi phí đọc bộ nhớ toàn cục (Global Memory latency) khi trích xuất descriptor không theo thứ tự liên tục (non-coalesced access).

#### 3. Tiêu chí hoàn thành (Definition of Done - DoD):
* [ ] Code C++ chạy đơn luồng đúng chuẩn, có comment trích dẫn rõ ràng.
* [ ] Đầu ra của Stage 2 & 3 phát hiện đúng và đủ các điểm đặc trưng rõ nét, ổn định trên các cặp ảnh thực nghiệm.
* [ ] Hoàn thành toàn bộ nội dung lý thuyết và thiết kế GPU của Stages 1, 2, 3 trong Notebook.

---

### THÀNH VIÊN 3: SYSTEM, DATA, PROFILING & NOTEBOOK LEAD
*(Phụ trách Bộ dữ liệu, Khung điều khiển, Đo đạc thực nghiệm, Định luật Amdahl & Báo cáo tổng thể)*

#### 1. Trách nhiệm về Dữ liệu (Dataset Preparation — Mục 3.6 Notebook):
* Chụp 4 cặp ảnh thực tế chất lượng cao bằng điện thoại/máy ảnh:
  * Đứng cố định một chỗ và xoay người (Rotate, don't walk) để không bị lỗi thị sai (parallax).
  * Độ phủ chồng lấp 30% - 50%.
  * 3 cặp chuẩn: 1 địa điểm trong nhà (indoor), 1 địa điểm ngoài trời (outdoor), 1 địa điểm tự chọn.
  * 1 cặp thử thách (challenging pair): bề mặt lặp lại, thiếu chi tiết hoặc ánh sáng yếu/ban đêm.
* Viết script Python (hoặc xử lý trực tiếp) cắt ảnh theo tỉ lệ **16:9** và xuất ra đúng 3 độ phân giải:
  * Small: $1280 \times 720$ (8 ảnh)
  * Medium: $1920 \times 1080$ (8 ảnh)
  * Large: $3840 \times 2160$ (8 ảnh)
  * $\rightarrow$ Đủ 24 ảnh chuẩn trong thư mục `dataset/`.
* Tải cặp ảnh nhân tạo **Synthetic pair** kèm ma trận chuyển đổi chuẩn (Ground-truth Homography) từ tài nguyên môn học.
* Tải bộ dữ liệu lên Google Drive, bật quyền truy cập công khai và lưu link vào tài liệu.
* **Trình bày Registered Sites trong Notebook (Mục 3.6):** Liệt kê và mô tả chi tiết từng địa điểm chụp (tên địa điểm, đặc điểm khung cảnh, lý do chọn, điều kiện ánh sáng) kèm ảnh minh họa trực quan các cặp ảnh gốc đã chụp.

#### 2. Trách nhiệm Khung chương trình (System Integration & Driver):
* Quản trị Git repository, thiết lập cấu trúc thư mục chuẩn `src/`, tạo `common.h`, `versions.h`, `main.cu`.
* Tích hợp thư viện đọc/ghi ảnh nhẹ `stb_image.h` và `stb_image_write.h` vào `src/stb_impl.cpp`.
* Ghép nối code của Thành viên 1 và Thành viên 2 thành hàm hoàn chỉnh trong `v0_<name>.cpp`.
* Hiện thực xử lý tham số dòng lệnh theo đúng chuẩn yêu cầu:
  `./stitch <left_image> <right_image> <output_image> --version 0 [--runs N]`.
* Tích hợp bộ đo thời gian bằng `std::chrono::high_resolution_clock` cho từng stage (loại trừ thời gian đọc ghi file ổ đĩa).

#### 3. Trách nhiệm Đo đạc (Profiling), Đánh giá & Hoàn thiện Báo cáo:
* **Kiểm tra tính đúng đắn (Correctness):** Chạy chương trình trên cặp ảnh Synthetic pair, tính sai số Homography Error so với ground-truth (chuẩn Frobenius norm hoặc khoảng cách chiếu điểm).
* **Xuất ảnh Panorama (bắt buộc — Mục 3.5 "Panoramas"):** Chạy V0 trên **tất cả 4 cặp ảnh dataset thực tế** (dùng độ phân giải Medium làm đại diện), lưu lại 4 ảnh panorama đầu ra và hiển thị trực quan trong Notebook — **mỗi địa điểm (site) phải có ít nhất 1 ảnh panorama kết quả** theo đúng yêu cầu đề bài.
* **Đo thời gian (Timing & Benchmarking):**
  * Chạy lặp lại ít nhất 5 lần (`--runs 5`), tính giá trị trung bình (Mean) và ghi nhận giá trị tối thiểu (Min) cho từng stage ở cả 3 kích thước ảnh (Small, Medium, Large).
  * Lập bảng thống kê chi tiết thời gian của từng giai đoạn (Per-stage timing table).
  * Vẽ biểu đồ cột / biểu đồ đường trực quan thể hiện thời gian thay đổi theo độ phân giải.
* **Phân tích Hot Stages & Định luật Amdahl:**
  * Xác định các giai đoạn chiếm nhiều thời gian nhất (thắt nút cổ chai).
  * Xác định tỷ lệ thời gian $p$ của các giai đoạn dự định tăng tốc bằng GPU.
  * Tính toán giới hạn tăng tốc lý thuyết: $\text{Speedup}_{\max} = \frac{1}{1 - p}$.
* **Viết nội dung Mục 3.4 (Sequential version V0) trong Notebook:**
  * Trình bày tổng quan kiến trúc mã nguồn V0: các file, hàm chính, cách các stage kết nối với nhau.
  * Trích dẫn rõ ràng tất cả các nguồn mã mở đã tham khảo/chuyển thể (tên dự án, link, giấy phép).
  * Giải thích ngắn gọn các thiết kế đáng chú ý (ví dụ: cấu trúc dữ liệu keypoint, cách lưu descriptor).
* **Viết Setup Cell (cell đầu tiên) trong Notebook:**
  * Viết cell tải/copy thư mục `src/` vào Colab session (từ Google Drive hoặc upload) và biên dịch chương trình. Ví dụ:
    ```python
    # Mount Drive, copy src/, compile
    from google.colab import drive
    drive.mount('/content/drive')
    !cp -r /content/drive/MyDrive/GroupXX/src /content/src
    !cd /content/src && nvcc -o stitch main.cu stb_impl.cpp v0_*.cpp -lm
    ```
  * Đây là điều kiện tiên quyết để Notebook chạy `Runtime -> Run all` thành công trên Colab mới.
* **Soạn thảo Notebook (`GroupXX_Midterm.ipynb`):**
  * Tích hợp Mục 3.1 (Thông tin nhóm & Kế hoạch làm việc), Mục 3.4 (Sequential version V0), Mục 3.5 (Kết quả V0 & Profiling), Mục 3.6 (Dataset).
  * Đảm bảo nguyên tắc: **Một nguồn sự thật duy nhất** (One source of truth - code biên dịch từ `src/`, KHÔNG copy code rời rạc vào notebook), **Khả năng tái lập** (Chạy từ đầu đến cuối với `Runtime -> Run all` trên Colab mới), **Số liệu từ output** (Toàn bộ số liệu trong bài viết đều xuất phát từ cell kết quả).
  * Đóng gói file nén `GroupXX_Midterm.zip` kiểm tra lần cuối trước khi nộp.

#### 4. Tiêu chí hoàn thành (Definition of Done - DoD):
* [ ] Đủ 24 ảnh dataset chất lượng cao + 1 cặp synthetic test + link Google Drive hoạt động tốt.
* [ ] Mô tả chi tiết từng registered site (tên, đặc điểm, lý do chọn) được trình bày trong Notebook mục 3.6.
* [ ] Chương trình V0 biên dịch mượt mà bằng lệnh `nvcc` hoặc `g++`, chạy CLI đúng chuẩn.
* [ ] Setup cell Colab (copy src/ + compile) hoạt động trơn tru trên session mới.
* [ ] Mục 3.4 trong Notebook trình bày rõ kiến trúc V0, trích dẫn đầy đủ nguồn mã mở.
* [ ] 4 ảnh panorama (1 per site) được xuất ra và hiển thị trong Notebook mục 3.5.
* [ ] Bảng đo đạc thời gian, tính toán Amdahl và đồ thị hiển thị đầy đủ, chính xác trong Notebook.
* [ ] Notebook Colab chạy thành công không có bất kỳ lỗi nào từ cell đầu đến cell cuối (`Runtime -> Run all`).

---

## 4. KẾ HOẠCH DÀI HẠN TỪ GIỮA KỲ ĐẾN BẢN NỘP CUỐI KỲ (FINAL SUBMISSION)

Bảng phân công này được thiết kế để sử dụng trực tiếp cho **Mục 3.1 trong Báo cáo giữa kỳ**:

| Phiên bản / Mốc thời gian | Thành viên 1 (Back-end Lead) | Thành viên 2 (Front-end Lead) | Thành viên 3 (System & Data Lead) |
| :--- | :--- | :--- | :--- |
| **Midterm (Tuần 7)**<br>*V0 - Sequential Baseline* | Hiện thực V0 cho Stages 4, 5, 6; đề xuất kế hoạch GPU cho Matching, RANSAC, Warping. | Hiện thực V0 cho Stages 1, 2, 3; đề xuất kế hoạch GPU cho Preprocessing, Detection, Description. | Thu thập dataset 24 ảnh; dựng khung `main.cu`; đo thời gian V0; phân tích Amdahl; hoàn thiện Notebook. |
| **V1: Basic CUDA (Tuần 8 - 9)**<br>*Chuyển toàn bộ pipeline sang GPU* | Chuyển Stages 4, 5, 6 sang các CUDA kernel cơ bản (1 match/thread, 1 pixel/thread). | Chuyển Stages 1, 2, 3 sang các CUDA kernel cơ bản (1 pixel/thread, trích xuất descriptor trên GPU). | Đo thời gian bằng CUDA Events cho từng kernel; kiểm tra tính nhất quán kết quả V1 so với V0; tính speedup. |
| **V2: Optimized CUDA (Tuần 10)**<br>*Tối ưu mức bộ nhớ & thuật toán* | Tối ưu Stage 4 & 5: Dùng Shared Memory Tiling cho Matching; dùng `__popc` cho Hamming distance; song song hóa RANSAC. | Tối ưu Stage 1, 2, 3: Separable convolution với Shared Memory Tiling + Halos; Prefix-Sum compaction cho keypoints. | Profiling chi tiết bằng Nsight Systems & Nsight Compute (Memory throughput, Occupancy, Warp stalls); cập nhật báo cáo. |
| **V3: Further Optimized (Tuần 11)**<br>*Tối ưu nâng cao & Pipeline* | Tối ưu Stage 6: Dùng CUDA Texture Memory cho phép nội suy song tuyến tính phần cứng; tối ưu blending song song. | Áp dụng CUDA Streams để chạy đồng thời Stages 1–3 cho cả 2 ảnh A và B trên 2 stream độc lập. | Tối ưu Kernel Fusion (gộp Grayscale + Blur); dùng Pinned Memory (Zero-copy) giảm độ trễ truyền dữ liệu PCIe. |
| **Final Submission (Tuần 12 - 13)**<br>*Đóng gói, Slide & Video* | Soạn slide phần Matching, Homography, Warping; thuyết minh phần kỹ thuật liên quan trong video demo. | Soạn slide phần Preprocessing, Feature Extraction; thuyết minh trong video demo. | Tổng hợp Notebook cuối kỳ, xuất file PDF, viết file `README.md`, dựng và upload video demo (20–30 phút) lên YouTube. |

---

## 5. QUY ƯỚC LÀM VIỆC NHÓM TRÊN GIT (GIT WORKFLOW)

Để quá trình làm việc nhóm diễn ra trơn tru, không xảy ra xung đột mã nguồn (merge conflict), nhóm tuân thủ các quy định sau:

### 1. Phân nhánh (Branching Strategy):
* `main`: Nhánh ổn định cao nhất, chỉ merge vào khi code đã được kiểm thử chạy đúng.
* `dev`: Nhánh tích hợp chính của cả nhóm.
* `feature/stage1-3-frontend`: Nhánh làm việc của Thành viên 2.
* `feature/stage4-6-backend`: Nhánh làm việc của Thành viên 1.
* `feature/system-benchmark`: Nhánh làm việc của Thành viên 3.

### 2. Quy chuẩn Commit Message:
Cú pháp: `[Phạm vi] Hành động: Mô tả ngắn gọn`
* `[v0-frontend] Feat: Hoàn thành FAST corner detector và NMS`
* `[v0-backend] Feat: Cài đặt RANSAC homography estimation với 1000 vòng lặp`
* `[system] Benchmark: Thêm bảng đo thời gian 5 lần lặp cho độ phân giải Large`
* `[docs] Update: Bổ sung phân tích định luật Amdahl vào notebook`

### 3. Quy tắc tích hợp:
* Tuyệt đối không commit trực tiếp các file binary (như file `.exe`, file chạy `./stitch`, các file ảnh kết quả nặng) lên repository.
* Sử dụng `.gitignore` để loại bỏ các tệp sinh ra tự động.

---

## 6. TIẾN ĐỘ THỰC HIỆN TỪNG NGÀY (SPRINT SCHEDULE)

* **Ngày 1 - 2:** Thiết lập Git, Thành viên 3 đi chụp ảnh và chuẩn bị 24 ảnh dataset. Thống nhất struct dữ liệu giữa Stage 3 và Stage 4.
* **Ngày 3 - 5:** Thành viên 1 và Thành viên 2 code độc lập các module của mình; test riêng lẻ từng hàm.
* **Ngày 6:** Thành viên 3 tích hợp code vào `main.cu` và `v0_<name>.cpp`, chạy thử toàn bộ pipeline trên 1 cặp ảnh nhỏ. Fix bug tương thích.
* **Ngày 7 - 8:** Chạy kiểm thử trên Synthetic Pair đo Homography Error; chạy benchmark đo thời gian trên 24 ảnh (Small, Medium, Large); Thành viên 3 tổng hợp bảng số liệu.
* **Ngày 9 - 10:** Cả 3 thành viên viết phần thuyết minh lý thuyết và kế hoạch GPU vào `GroupXX_Midterm.ipynb`.
* **Ngày 11:** Test chạy Colab từ đầu đến cuối trên session mới (`Runtime -> Run all`). Đóng gói `GroupXX_Midterm.zip` và nộp bài.
