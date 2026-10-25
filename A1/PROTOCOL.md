# Protocol A1.1

Trạng thái: kế hoạch cho A1.2, không phải kết quả đã chạy.

## Dữ liệu
PathMNIST 28×28 RGB, 9 lớp; train 89996, validation 10004, test 7180. License CC BY 4.0. Nguồn https://medmnist.com/ ; file chính thức https://zenodo.org/records/10519652/files/pathmnist.npz?download=1 . MD5 dự kiến a8b06965200029087d5bd730944a56c1. Giữ nguyên split. Chín lớp thấp hơn mức ưu tiên ≥10 của đề; giải thích trong báo cáo. Không đổi dataset sau A1.1 nếu chưa được phép theo đề.

## Mô hình và preprocessing dự kiến
- CNN: torchvision ResNet-18, ResNet18_Weights.IMAGENET1K_V1.
- ViT: google/vit-base-patch16-224, đầu ra 9 lớp.
- Resize trực tiếp 224×224, bilinear; train lật ngang p=0.5; val/test không augmentation ngẫu nhiên.
- ResNet: mean [0.485,0.456,0.406], std [0.229,0.224,0.225].
- ViT: mean [0.5,0.5,0.5], std [0.5,0.5,0.5]; kiểm tra lại processor của checkpoint khi hiện thực.
- Dùng cùng preprocessing cho bản scratch tương ứng. Không gọi normalization khác nhau là hoàn toàn cùng đầu vào.
- Nguồn và recipe pretraining khác nhau có thể ảnh hưởng so sánh; không quy mọi chênh lệch cho kiến trúc.

## Ma trận thí nghiệm
| Chế độ | ResNet-18 | ViT-Base/16 | LR dự kiến |
|---|---|---|---|
| Scratch | Ngẫu nhiên, train toàn bộ | Ngẫu nhiên, train toàn bộ | 1e-4 |
| Pretrained head | fc | classifier | 1e-3 |
| Pretrained partial | layer4 + fc | 2 encoder block cuối + layernorm cuối + classifier | 1e-5 |
| Pretrained full | Toàn bộ | Toàn bộ | 1e-5 |

AdamW, weight_decay=0.01, CrossEntropyLoss, LR cố định; batch=2, dự kiến 10 epochs/cấu hình; AMP trên CUDA; num_workers=0. Seed=42 cho random, NumPy, torch và DataLoader. Ngân sách 10 epochs chưa bảo đảm hội tụ. Nếu cần thay đổi chỉ dùng validation và lưu cấu hình mới trước đánh giá test.

Với head-only và partial CNN, các module backbone bị freeze giữ ở eval để BatchNorm không cập nhật running statistics. Kiểm tra requires_grad, gradient và số tham số trainable cho từng chế độ. Scratch không nạp pretrained.

## Đánh giá
Chọn validation Macro-F1 cao nhất; tie-break validation loss thấp hơn. F1 trên đủ 9 lớp với zero_division=0. Sau chốt cấu hình đánh giá toàn bộ test: Accuracy, Macro-F1, confusion matrix, lỗi điển hình. Ghi thời gian train tách tải model và VRAM cấp phát bằng cùng phương pháp; số tham số tổng/trainable. Không lấy test để chọn LR, epochs hoặc checkpoint.

## Đã làm và chưa làm
Đã kiểm tra ViT 20 batch train/10 batch validation. Chưa chạy đủ ma trận hoặc hoàn thành tái lập paper; không dùng phép thử để so sánh với kết quả paper. Notebook riêng PathMNIST_A1_1.ipynb chỉ dùng kiểm tra A1.1, chưa có trainer cho A1.2. YOLO Detection chưa được hoàn thành trong phần này.

