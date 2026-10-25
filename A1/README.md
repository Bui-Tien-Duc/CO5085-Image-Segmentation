# A1 — CNN vs Vision Transformer on PathMNIST

Sinh viên: Bùi Tiến Đức  
MSSV: 2680018  
Môn học: CO5085

## Dataset

- PathMNIST: 107.180 ảnh RGB, 9 lớp.
- Train: 89.996; validation: 10.004; test: 7.180.
- Giữ nguyên split được cung cấp.
- Ảnh 28 × 28, resize lên 224 × 224 khi đưa vào mô hình.
- Nguồn: https://medmnist.com/
- License: CC BY 4.0.
- Dữ liệu đặt tại `A1/data/pathmnist.npz`, không đưa lên GitHub.
- Dataset có 9 lớp, thấp hơn mức ưu tiên ≥ 10 lớp của đề.
  Lựa chọn nhằm gắn với paper tái lập và có quy mô dữ liệu lớn.

## Paper nền

Halder et al. (2024).
Implementing vision transformer for classifying 2D biomedical images.
Scientific Reports, 14, 12567.

https://doi.org/10.1038/s41598-024-63094-9

Code tác giả: https://github.com/arindam369/MedMNIST-ViT
Giữ lại LICENSE của repository tác giả trong thư mục mã nguồn.

## Môi trường đã kiểm tra

- Windows, Python 3.11.
- PyTorch 2.11.0+cu128.
- Transformers 4.57.6.
- Ultralytics 8.4.188.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- Danh sách thư viện: requirements-lock.txt.

## Kế hoạch thí nghiệm

- CNN: ResNet-18.
- Transformer: ViT-Base/16.
- Mỗi mô hình: from-scratch, pretrained head-only,
  pretrained partial fine-tune và pretrained full fine-tune.
- Batch size dự kiến: 2; epochs dự kiến: 10.
- Optimizer: AdamW; loss: CrossEntropyLoss.
- LR dự kiến: scratch 1e-4; head-only 1e-3;
  partial/full fine-tune 1e-5.
- Partial: ResNet layer4 + fc; ViT hai encoder block cuối,
  layernorm cuối + classifier.
- Chọn checkpoint theo validation Macro-F1.
- Báo cáo Accuracy, Macro-F1, confusion matrix,
  thời gian, VRAM và số tham số trainable.
- Xác nhận preprocessing của từng checkpoint trước khi train chính thức.

## Trạng thái A1.1

Đã kiểm tra GPU, dữ liệu, EDA và DataLoader.
ViT đã chạy thử 20 batch train và 10 batch validation.
Đây là kiểm tra pipeline, chưa phải kết quả so sánh A1.2.

## Hướng dẫn kiểm tra

1. Tải PathMNIST và đặt tại `A1/data/pathmnist.npz`.
2. Mở notebook `MedMNIST-ViT-main/PathMNIST.ipynb`.
3. Chọn kernel môi trường A1.
4. Các ô bổ sung hiện dùng đường dẫn máy cá nhân:
   sửa `data_path` theo máy trước khi chạy.
5. Chạy lần lượt các ô kiểm tra môi trường, EDA,
   DataLoader và pipeline; chưa dùng Run All.

## Minh chứng

- results/class_distribution.csv
- results/sample_images.png
- results/train_class_distribution.png
- results/vit_pipeline_check.json