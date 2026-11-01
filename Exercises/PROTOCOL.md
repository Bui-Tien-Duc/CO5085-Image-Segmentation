# Protocol dự kiến cho E1–E3

Trạng thái: kế hoạch, chưa phải cấu hình đã chạy hoặc đã được giảng viên duyệt.

- Dataset dự kiến: Fashion-MNIST; dùng cùng dataset cho cả ba bài.
- Giữ 10.000 ảnh test chính thức riêng. Chia 60.000 train chính thức thành 54.000 train và 6.000 validation; seed 42; lưu chỉ số split để tái lập. Kiểm tra phân bố lớp.
- Ảnh xám 28 × 28, không resize lên 224. Chọn normalization từ train và ghi lại; không thống kê bằng test.
- Ngân sách dự kiến: 10 epochs, batch 64; smoke test trước, ghi cấu hình thực tế. Không coi đã đủ hội tụ.
- Chọn checkpoint bằng validation accuracy cao nhất; hòa thì validation loss thấp hơn.
- Metric: accuracy; bổ sung Macro-F1, confusion matrix, thời gian và số tham số.
- Tự viết loop forward → loss → backward → optimizer.step. CrossEntropyLoss nhận logits; không softmax trước loss.
- E2 dự kiến patch 7 × 7 (16 token) và row (28 token), embedding 64, heads 4, blocks 2.
- Kiểm tra MSA custom so với PyTorch bằng cùng trọng số/input và tắt dropout trước khi so sánh chất lượng training.
- E3 dự kiến row-wise: 28 bước, mỗi bước 28 giá trị. Giữ split để so với E1.
- Mở rộng chọn bằng validation; không hứa biến thể tối ưu sẽ tốt hơn baseline.
- Lưu last checkpoint và optimizer nếu cần resume. Không đưa môi trường ảo, dữ liệu thô hoặc toàn bộ runs lên Git.
