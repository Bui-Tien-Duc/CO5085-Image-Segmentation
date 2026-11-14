# E1 — Fashion-MNIST

Bùi Tiến Đức · MSSV 2680018 · CO5085

## Trạng thái

Đã hoàn thành thực nghiệm Softmax, MLP, CNN và MLP có Dropout 0,5. Mỗi cấu hình 10 epochs, seed 42; checkpoint chọn bằng validation Accuracy cao nhất, lấy epoch đầu khi hòa. Báo cáo chung E1–E3 chưa hoàn thành.

## Dataset và cấu hình

Fashion-MNIST grayscale 28 × 28, 10 lớp. Split 54.000 train / 6.000 validation / 10.000 test, seed 42. Normalization từ train: mean 0.2858857184540482, std 0.3529602921170978. Batch 64, Adam, LR 0.001, weight decay 0, CrossEntropyLoss trên logits. Không dùng test để chọn checkpoint.

## Kết quả test

| Cấu hình | Tham số | Best epoch | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Softmax | 7.850 | 9 | 84,23% | 84,14% |
| MLP | 235.146 | 9 | 88,40% | 88,31% |
| CNN | 206.922 | 8 | 91,28% | 91,26% |
| MLP + Dropout 0,5 | 235.146 | 9 | 87,31% | 87,14% |

CNN hơn MLP 2,88 điểm phần trăm Accuracy, với ít tham số hơn. Dropout 0,5 thấp hơn MLP không Dropout 1,09 điểm trong thiết lập này. Một seed chưa đủ kết luận tổng quát.

CNN có Shirt F1 74,83%, recall 72,10%; nhầm 106 Shirt thành T-shirt/top và 80 Coat thành Pullover. Test #23: Ankle boot bị dự đoán Sandal, confidence 0,992. Confidence cao không bảo đảm đúng.

## Minh chứng

- [HTML có hình](https://bui-tien-duc.github.io/CO5085-Image-Segmentation/E1/results/evaluation_final/E1_report.html)
- [Bảng CSV](results/evaluation_final/comparison.csv)
- [PDF E1 tạm thời](../reports/E1_Bui_Tien_Duc_2680018.pdf)
- [Release checkpoint](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/tag/e1-fashionmnist-v1)

## Checkpoint

Các file đã công bố có SHA256 khớp test_metrics.json đã kiểm tra.

- [softmax_best.pt](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/download/e1-fashionmnist-v1/softmax_best.pt) → đặt thành E1/results/softmax_v1/best.pt
- [mlp_best.pt](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/download/e1-fashionmnist-v1/mlp_best.pt) → đặt thành E1/results/mlp_v1/best.pt
- [cnn_best.pt](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/download/e1-fashionmnist-v1/cnn_best.pt) → đặt thành E1/results/cnn_v1/best.pt
- [mlp_dropout05_best.pt](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/download/e1-fashionmnist-v1/mlp_dropout05_best.pt) → đặt thành E1/results/mlp_dropout05_v1/best.pt

Giữ config.json, history.csv và summary.json trong từng run. Không dùng last.pt thay best.pt để đánh giá.

## Chạy lại từ gốc repository

Chuẩn bị dữ liệu: ` .\\A1\\.venv\\Scripts\\python.exe .\\Exercises\\prepare_data.py `

Đánh giá bốn checkpoint, không train:

```powershell
.\\A1\\.venv\\Scripts\\python.exe .\\E1\\evaluate_e1.py --output .\\E1\\results\\evaluation_recheck
```

Output phải là thư mục mới. Không cần chạy lại để xem kết quả đã công bố.

## Paper, phạm vi và hạn chế

N. Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” JMLR, 15(56), 1929–1958, 2014. [Bản xuất bản](https://www.jmlr.org/papers/v15/srivastava14a.html).

Áp dụng cơ chế Dropout vào MLP Fashion-MNIST là tái hiện có giới hạn, chưa tái lập toàn bộ paper. Chưa đính kèm minh chứng chỉ mục. Chỉ một seed và p=0,5; không tối ưu theo kết quả test. Train accuracy đo ở train mode, validation ở eval mode nên không so trực tiếp khoảng cách hai chỉ số để kết luận về Dropout. AI hỗ trợ chuẩn bị/rà soát code và tài liệu; người thực hiện kiểm tra, hiểu và chịu trách nhiệm về kết quả.

