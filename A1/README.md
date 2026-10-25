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
ViT và ResNet-18 đã chạy thử; mỗi mô hình 20 batch train và 10 batch validation.
ResNet-18 sử dụng DataLoader riêng với normalization ImageNet; ViT dùng normalization của checkpoint tương ứng.
Đây là kiểm tra pipeline, chưa phải kết quả so sánh A1.2.

## Hướng dẫn cài đặt và chạy lại

Chạy PowerShell từ thư mục A1. Nếu đã cài đúng môi trường, không cần cài lại.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-a1.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyterlab
```

Chọn kernel A1. Mở **PathMNIST_A1_1.ipynb**, chạy theo thứ tự; notebook này tách riêng khỏi các ô và output gốc tác giả. Dùng đường dẫn tương đối, chạy từ A1 hoặc gốc repo.

Tải [pathmnist.npz](https://zenodo.org/records/10519652/files/pathmnist.npz?download=1) vào A1/data. Kiểm tra checksum từ gốc repo:

```powershell
Get-FileHash .\A1\data\pathmnist.npz -Algorithm MD5
```

MD5 dự kiến: a8b06965200029087d5bd730944a56c1. Không đưa dữ liệu hoặc .venv lên GitHub.

requirements-lock.txt là bản chụp môi trường Windows đã dùng, có cả YOLO và các phụ thuộc khác. Nó không chứa cấu hình index tải CUDA; không coi pip install -r requirements-lock.txt từ PyPI mặc định là hướng dẫn đầy đủ. requirements-a1.txt là danh sách chọn từ phiên bản môi trường đã ghi; chưa kiểm thử cài mới ở máy khác.

## Protocol và báo cáo

- [Protocol chi tiết](PROTOCOL.md)
- [PDF A1.1](../reports/A1_1_Bui_Tien_Duc_2680018.pdf)
- [Website A1](https://bui-tien-duc.github.io/CO5085-Image-Segmentation/#a1)

## Minh chứng đã chạy

- [Thống kê lớp](results/class_distribution.csv)
- [Ảnh mẫu](results/sample_images.png)
- [Phân bố train](results/train_class_distribution.png)
- [Kiểm tra pipeline ViT](results/vit_pipeline_check.json)
- [Kiểm tra pipeline ResNet-18](results/resnet18_pipeline_check.json)
- [Notebook A1.1](PathMNIST_A1_1.ipynb)

Notebook đã được chạy và lưu minh chứng kiểm tra pipeline. Các file JSON chỉ mô tả phép thử trên tập nhỏ, không phải Accuracy cuối hay kết quả tái lập đầy đủ paper. Chạy lại notebook sẽ cập nhật JSON và EDA trên máy. Không dùng test để chọn cấu hình hoặc checkpoint.

## Phần còn lại

A1.2 chưa hoàn thành: hiện thực và huấn luyện tám cấu hình theo protocol, lưu log/checkpoint, chọn checkpoint bằng validation Macro-F1 và đánh giá toàn bộ test để so sánh CNN–Transformer.

Phạm vi tái lập tập trung vào PathMNIST; các cấu hình ResNet-18 và freeze là phần thí nghiệm mở rộng, không được coi là đã tái lập toàn bộ paper.

Detection YOLO là phần thực hành riêng, chưa có minh chứng fine-tune trong A1.

