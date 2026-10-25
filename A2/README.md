# CO5085 – Semantic Image Segmentation

## 1. Giới thiệu

Bài thực hành xây dựng hệ thống phân đoạn ngữ nghĩa ảnh giao thông đô thị bằng mô hình U-Net trên bộ dữ liệu CamVid.

## 2. Bộ dữ liệu

Nguồn tải CamVid: https://github.com/stonewst/CamVid-for-Segmentation

Dữ liệu sử dụng gồm 367 ảnh train, 101 ảnh validation và 233 ảnh test, với 11 lớp ngữ nghĩa.

Sau khi tải, giải nén dữ liệu vào `A2/data/CamVid/`.

## 3. Môi trường

- Python 3.11
- PyTorch hỗ trợ CUDA
- GPU NVIDIA RTX 3050 Ti Laptop GPU

## 4. Chạy chương trình
Các lệnh bên dưới được chạy tại thư mục A2.
Đường dẫn .venv áp dụng khi môi trường ảo nằm trong A2.

Kiểm tra dữ liệu:

`python check_camvid.py`

Huấn luyện:

`python train.py --epochs 10 --batch-size 2 --size 256`


Đánh giá checkpoint trên toàn bộ tập test:

```powershell
.\.venv\Scripts\python.exe .\verify_checkpoint.py
```

Tạo hình so sánh ảnh đầu vào, nhãn thật và dự đoán:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py
```

Trực quan hóa dữ liệu và biểu đồ:

`python show_camvid.py`

`python plot_history.py`

## 5. Kết quả thực nghiệm

U-Net được huấn luyện trên CamVid trong 10 epochs,
batch size 2, kích thước ảnh 256 × 256.

- Optimizer: AdamW.
- Learning rate: 0.001.
- Loss: CrossEntropyLoss, ignore_index=255.
- Best validation mIoU: 0.4434 tại epoch 9.
- Test mIoU: 0.3806.
- Test Mean Dice: 0.4469.
- Test loss: 0.5489.

Checkpoint được chọn theo validation mIoU cao nhất.
Đã đánh giá lại checkpoint trên toàn bộ tập test;
kết quả chi tiết lưu tại results/test_metrics_verified.txt.

Hình minh họa:
- results/prediction.png
- results/training_curves.png

## 6. Báo cáo

Báo cáo đầy đủ được lưu trong thư mục `reports/` của repository.

## 7. Tài liệu tham khảo

- Ronneberger et al. (2015), U-Net.
- Brostow et al. (2009), CamVid.
- Zhou et al. (2020), UNet++.
