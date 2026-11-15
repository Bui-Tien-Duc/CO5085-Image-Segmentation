# CO5085 – Semantic Image Segmentation

## 1. Giới thiệu

Bài thực hành xây dựng hệ thống phân đoạn ngữ nghĩa ảnh giao thông đô thị bằng hai mô hình U-Net và UNet++ trên bộ dữ liệu CamVid.

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

.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

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

Báo cáo được lưu trong dự án. Bản PDF cần cập nhật kết quả A2.2 và phân tích paper trước khi chốt hồ sơ.

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

Phạm vi thực nghiệm: so sánh hai kiến trúc trên cùng cách chia
CamVid, kích thước ảnh và ngân sách huấn luyện;
đánh giá bằng mIoU và Mean Dice.

## Kết quả A2.2

Đã huấn luyện và đánh giá U-Net và UNet++ trên CamVid.
Tập dữ liệu gồm 367 ảnh train, 101 ảnh validation và 233 ảnh test.

| Mô hình | Best epoch | Test mIoU | Test mean Dice | Test loss |
|---|---:|---:|---:|---:|
| U-Net | 9 | 38,06% | 44,69% | 0,548879 |
| UNet++ | 10 | 39,54% | 45,49% | 0,504930 |

Trong thí nghiệm hiện tại, UNet++ tăng khoảng 1,48 điểm phần trăm
mIoU và 0,80 điểm phần trăm mean Dice so với U-Net.

Phạm vi: ảnh 256×256, batch size 2, 10 epochs, seed 42;
AdamW, learning rate 0.001, weight decay 0.01;
CrossEntropyLoss với ignore_index=255, không augmentation.
Checkpoint được chọn theo validation mIoU cao nhất, không theo test.
Bản UNet++ dùng base 16, một đầu ra và không deep supervision;
có 2.292.139 tham số. Đây là bản điều chỉnh cho CamVid,
chưa tái lập đầy đủ paper gốc.

UNet++ đạt validation mIoU 43,09% tại epoch 10;
test pixel accuracy 84,92% trên đủ 233 ảnh test.
Hạn chế: Pole, Pedestrian và Bicyclist có test IoU bằng 0;
SignSymbol và Fence có IoU gần 0. Pixel accuracy không phản ánh
đầy đủ chất lượng phân đoạn từng lớp. Kết quả một seed và 10 epochs
chưa đủ để khẳng định ưu thế tổng quát của UNet++.

Kết quả chi tiết:

- [U-Net đã xác minh](results/test_metrics_verified.txt).
- [UNet++ test metrics](results/evaluation_unetpp_v1/test_metrics.json).
- [IoU/Dice từng lớp](results/evaluation_unetpp_v1/per_class_metrics.csv).
- [Confusion matrix](results/evaluation_unetpp_v1/confusion_matrix.csv).
- [Ảnh dự đoán](results/evaluation_unetpp_v1/sample_predictions.png).
- [Lịch sử train UNet++](results/unetpp_v1/history.csv).
- [Cấu hình UNet++](results/unetpp_v1/config.json).

SHA256 checkpoint UNet++ đã đánh giá:

```text
2a9247457a163591212fa4357dcf93c43ee44104cb5ad9683e68898e974b0abb
```

### Chạy UNet++ bằng môi trường A1

Các lệnh sau chạy tại thư mục gốc repository, sử dụng môi trường
`A1/.venv` đã có. Không cần tạo lại môi trường nếu đã chạy thành công.

```powershell
.\A1\.venv\Scripts\python.exe -u .\A2\check_unetpp.py
.\A1\.venv\Scripts\python.exe -u .\A2\train_unetpp.py --data .\A2\data\CamVid --epochs 10 --batch-size 2 --output .\A2\results\unetpp_v1
```

Nếu đã hoàn thành lượt train thì không chạy lại. Nếu bị ngắt,
chạy cùng lệnh train và thêm `--resume`; epoch đang dở sẽ chạy lại
sau epoch cuối đã lưu trong `last.pt`.

Đánh giá checkpoint đã chọn, không huấn luyện thêm:

```powershell
.\A1\.venv\Scripts\python.exe -u .\A2\evaluate_unetpp.py --data .\A2\data\CamVid --checkpoint .\A2\results\unetpp_v1\best.pt --output .\A2\results\evaluation_unetpp_v1
```

Trạng thái hồ sơ: thực nghiệm hai mô hình và video demo đã có;
phân tích paper đầy đủ, PDF và website cần được cập nhật đồng bộ.


Áp dụng UNet++ trên CamVid chưa phải tái lập đầy đủ
thí nghiệm ảnh y sinh trong paper gốc.

## Video demo

[Video demo A2 – U-Net và UNet++ trên CamVid](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/tag/Video-a2-camvid-v1)

Trong mục Assets, tải file
`A2_Demo_UNet_UNetPP_Bui_Tien_Duc_2680018.mp4`.

Video minh họa nạp checkpoint, đánh giá trên 233 ảnh test,
xem hình phân đoạn và thảo luận kết quả.
