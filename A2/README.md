# CO5085 – Semantic Image Segmentation

## 1. Giới thiệu

Bài thực hành xây dựng hệ thống phân đoạn ngữ nghĩa ảnh giao thông đô thị bằng mô hình U-Net trên bộ dữ liệu CamVid.

## 2. Bộ dữ liệu

Nguồn tải CamVid: https://github.com/stonewst/CamVid-for-Segmentation

Dữ liệu sử dụng gồm 367 ảnh train, 101 ảnh validation và 233 ảnh test, với 11 lớp ngữ nghĩa.

Sau khi tải, giải nén dữ liệu vào `A2/data/CamVid/`.

## 3. Môi trường

### Chuẩn bị môi trường

Sử dụng Python 3.11. Tại thư mục A2, tạo môi trường:

```powershell
py -3.11 -m venv .venv
```

Cài các thư viện theo phiên bản đã ghi nhận:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Bản PyTorch trong thực nghiệm là 2.11.0+cu128.
Nếu pip không tìm thấy bản này trên nguồn mặc định, cài PyTorch
từ nguồn wheel CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

### Chuẩn bị dữ liệu

Tải CamVid từ nguồn đã dẫn và sắp xếp như sau:

```text
data/CamVid/
├── class_palette.csv
├── train/
├── trainannot/
├── val/
├── valannot/
├── test/
└── testannot/
```

Mask có tên tương ứng dạng <tên ảnh>_L.png.
class_palette.csv cần có các cột name, r, g, b.

### Đánh giá checkpoint

Checkpoint được cung cấp tại results/best_unet.pt.
Chạy tại thư mục A2:

```powershell
.\.venv\Scripts\python.exe .\verify_checkpoint.py
```

Kết quả tham chiếu trên 233 ảnh test:
loss khoảng 0.5489, mIoU khoảng 0.3806, Mean Dice khoảng 0.4469.
Sai lệch nhỏ có thể xuất hiện giữa các môi trường.

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

## A2.1 Đề tài và kế hoạch nghiên cứu

Đề tài: So sánh U-Net và UNet++ cho phân đoạn ngữ nghĩa
ảnh giao thông đô thị trên CamVid.

Paper lựa chọn:
Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang,
“UNet++: Redesigning Skip Connections to Exploit Multiscale
Features in Image Segmentation,” IEEE Transactions on Medical
Imaging, vol. 39, no. 6, pp. 1856–1867, 2020.
DOI: 10.1109/TMI.2019.2959609.

Phạm vi dự kiến: so sánh hai kiến trúc trên cùng cách chia
CamVid, kích thước ảnh và ngân sách huấn luyện;
đánh giá bằng mIoU và Mean Dice.

Hiện trạng: đã hoàn thành baseline U-Net và kiểm chứng
checkpoint. UNet++ và phần tối ưu hóa chưa thực hiện.

Áp dụng UNet++ trên CamVid chưa phải tái lập đầy đủ
thí nghiệm ảnh y sinh trong paper gốc.
