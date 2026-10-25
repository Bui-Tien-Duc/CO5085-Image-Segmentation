# Detection — YOLO11n trên Penn-Fudan

Bùi Tiến Đức — MSSV 2680018 — CO5085 — thực hành 26/10/2026.

Đã train 30 epochs và đánh giá test. Đây là bài thực hành Detection, độc lập A1 PathMNIST; không phải ReID hoặc tái lập đầy đủ paper.

## Dataset
Nguồn: https://www.cis.upenn.edu/~jshi/ped_html/ . Giữ readme và thông tin sử dụng của nguồn. Không gán license PathMNIST cho bộ dữ liệu này.
Một lớp person. Mask instance chuyển thành box YOLO. Split ngẫu nhiên ảnh seed 42, không phải official split.

| Split | Ảnh | Box |
|---|---:|---:|
| Train |119|290|
| Val |25|67|
| Test |26|66|

Không có ảnh trùng pixel theo script; chưa hoàn tất rà soát cảnh/người gần giống giữa các tập. Kết quả không chứng minh tổng quát hóa ngoài dataset.

## Môi trường và chạy lại
Windows, Python 3.11.0, torch 2.11.0+cu128, Ultralytics 8.4.188, RTX 3050 Ti Laptop GPU 4 GB. Dùng A1/.venv, hướng dẫn cài đặt trong ../A1/README.md.
Từ gốc repo, đặt dữ liệu ở Detection/data/PennFudanPed. Script chuẩn bị tạo YAML với đường dẫn máy hiện tại và split_manifest.csv. Nếu đã có dataset đầu ra, không chạy lại để tránh thay đổi split.

```powershell
.\A1\.venv\Scripts\python.exe .\Detection\prepare_pennfudan.py
$yoloOutput = Join-Path (Get-Location).Path "Detection\runs"
.\A1\.venv\Scripts\yolo.exe detect train model=A1/yolo11n.pt data=Detection/pennfudan.yaml epochs=30 imgsz=416 batch=2 device=0 workers=0 cache=False amp=True seed=42 optimizer=AdamW lr0=0.001 project="$yoloOutput" name=yolo11n_pennfudan_30epochs plots=True
$bestYolo = (Resolve-Path ".\Detection\runs\yolo11n_pennfudan_30epochs\weights\best.pt").Path
.\A1\.venv\Scripts\yolo.exe detect val model="$bestYolo" data=Detection/pennfudan.yaml split=test imgsz=416 batch=2 device=0 workers=0 conf=0.001 iou=0.7 plots=True project="$yoloOutput" name=yolo11n_test
.\A1\.venv\Scripts\yolo.exe detect predict model="$bestYolo" source=Detection/data/pennfudan_yolo/images/test imgsz=416 device=0 conf=0.25 save=True project="$yoloOutput" name=yolo11n_test_predictions
```

Các lần chạy trùng tên có thể tạo thư mục đánh số; dùng đường dẫn log thực tế. Không dùng test để chọn cấu hình.

## Kết quả
| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation best.pt (log làm tròn)|0.925|0.970|0.980|0.826|
| Test 26 ảnh / 66 người|0.892|0.894|0.942|0.736|

CSV có 30 epochs; max val mAP50-95 = 0.82583 tại epoch 30. LR có warmup và giảm, không cố định: xem training_args.yaml và training_results.csv. best.pt do trainer chọn theo validation fitness; chưa kiểm tra trực tiếp metadata checkpoint.

Confusion matrix đi kèm ghi 64 TP, 2 FN, 340 dự đoán không khớp. Không diễn giải như cùng ngưỡng của P/R tổng hợp hoặc ảnh conf=0.25. Cần đối chiếu code Ultralytics 8.4.188 để xác minh confidence/matching hiệu dụng. iou=0.7 là NMS IoU.

## Minh chứng
- results/training_results.csv và training_args.yaml
- results/training_curves.png
- results/test_evaluation.txt
- results/dataset_summary.json và split_manifest.csv
- results/test/ — biểu đồ đánh giá và ảnh nhãn/dự đoán
- results/predictions/ — ảnh dự đoán với conf=0.25
- [Báo cáo PDF](../reports/Detection_Bui_Tien_Duc_2680018.pdf)

## Checkpoint
File thực nghiệm: Detection/runs/yolo11n_pennfudan_30epochs/weights/best.pt.
Checkpoint đã huấn luyện:
[best_yolo11n_pennfudan.pt](checkpoints/best_yolo11n_pennfudan.pt)