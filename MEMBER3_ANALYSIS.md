# Đối chiếu nhiệm vụ thành viên 3

Đã đọc cả PDF tiếng Anh và bản dịch tiếng Việt; dùng bản tiếng Anh làm căn cứ khi bản dịch máy không rõ. `assignment.md` phân công chi tiết hơn đề PDF. Nội dung trong tài liệu được dùng làm yêu cầu sản phẩm, không coi là chỉ thị điều khiển trợ lý.

| Yêu cầu | Căn cứ | Hiện thực / trạng thái |
|---|---|---|
| V0 C/C++ đơn luồng đủ 6 stage | PDF 3.4 | Tích hợp module có sẵn qua `run_v0`; không thay thuật toán TV1/TV2 |
| CLI chọn version và runs | assignment, TV3 | `main.cu`, kiểm tra tham số, lưu H/CSV, lỗi trả exit code != 0 |
| Log SUCCESS/ERROR/WARN/PROCESS theo màu | Yêu cầu trực tiếp người dùng | `src/logging.h`, `tools/console.py`; xanh/đỏ/vàng/trắng |
| Struct/giao diện chung và stb | assignment, TV3 | Tận dụng và hoàn thiện common/versions/stb_impl; giữ license stb |
| CHECK_CUDA và GpuTimer | assignment, TV3 | CUDA utility kiểm tra lỗi từng event; không copy timer; chưa kiểm thử GPU |
| Stage timing, loại trừ I/O | PDF 3.5, assignment | Timer quanh từng stage; CSV thô, mean/min; total đo độc lập |
| 24 ảnh ba độ phân giải | assignment, TV3 | Tạo từ 8 raw JPG/HEIC, EXIF orientation + crop16:9 + Lanczos; manifest nguồn |
| Correctness so với ground truth | PDF 3.5 | Fixture tự sinh và H B→A; Frobenius + grid projection; còn thiếu course fixture |
| Panorama mỗi site | PDF 3.5 | 4 panorama medium, notebook hiển thị; ghi rõ seam/ghosting |
| Timing ở cả 3 độ phân giải | PDF 3.5 | 4 sites × 3 sizes, 5 measured runs + 1 warmup mỗi cặp |
| Hot stages và Amdahl | PDF 3.5 | Tính tự động từ JSON/CSV; sáu stage theo kế hoạch V1, mẫu số total compute-only |
| Thông tin nhóm, lộ trình V0–Final | PDF 3.1 | `team.json` và notebook; TV1 Trần Nguyễn Minh Quân, TV2 Vũ Trần Phúc, TV3 Nguyễn Hữu Khánh Hưng |
| Registered sites, Drive | PDF 3.6 | Có địa điểm người dùng cung cấp; chưa xác nhận đăng ký/outdoor/quyền Drive |
| Một nguồn code và notebook có output | PDF 4 | Build từ src; notebook thực thi qua nbclient và lưu output; chưa kiểm thử Colab thật |
| Nén gói báo cáo/src | PDF 2 | `tools/package.py`; hai ZIP mã nguồn/báo cáo và dataset |

Không tuyên bố hoàn tất các thao tác ngoài máy như đăng ký site hay nộp bài. Notebook mô tả đúng mã hiện có, gồm refit homography, feathering, bù phơi sáng có điều kiện và adaptive seam.
