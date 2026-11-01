# E1 — Phân loại ảnh với Softmax, MLP và CNN

Bùi Tiến Đức · MSSV 2680018 · CO5085

## Trạng thái

Đã tạo khung thư mục và kế hoạch. Chưa có code thực nghiệm, training, checkpoint hoặc kết quả.
Hạn nộp: **23:59 ngày 02/11/2026, GMT+7**.

## Dataset

Dự kiến Fashion-MNIST, dùng chung cho E1–E3. Chưa tải hoặc xác nhận dữ liệu trong mục này.
Giữ split chung, seed và tiền xử lý được công bố ở [protocol](../Exercises/PROTOCOL.md).

## Công việc

- [ ] Softmax trên ảnh flatten.
- [ ] MLP trên ảnh flatten.
- [ ] CNN cho ảnh nhỏ.
- [ ] Tự viết training loop PyTorch, đánh giá validation/test.
- [ ] Bảng accuracy, số tham số, đường cong và phân tích lỗi.
- [ ] Mở rộng dự kiến: MLP có/không dropout; chọn p bằng validation.

## Paper dự kiến

*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 2014.

[Bản xuất bản](https://www.jmlr.org/papers/v15/srivastava14a.html). Chưa ghi đã tái lập hoặc tối ưu thành công.
Đây là phần mở rộng bên cạnh các yêu cầu bắt buộc của bài E. Phạm vi, phương trình, code tham khảo và điều kiện sử dụng phải được xác nhận trước khi hiện thực.

## Cách chạy

Chưa có lệnh train/evaluate. Sẽ cập nhật khi code được kiểm tra. Có thể dùng môi trường A1/.venv đang hoạt động; chưa yêu cầu cài thêm thư viện.

## Báo cáo

Dùng **một báo cáo PDF chung** cho E1–E3, dự kiến `reports/exercises.pdf`. Hiện file chưa được tạo. Xem [khung báo cáo](../Exercises/REPORT_OUTLINE.md).

## Tái lập và công bố

Lưu config, seed, split, log, checkpoint chọn bằng validation và kết quả test thật. Không dùng test để chọn cấu hình. Ghi nguồn code và việc sử dụng AI theo quy định đề bài; cần hiểu và bảo vệ được phần hiện thực.
