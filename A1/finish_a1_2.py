"""Evaluate saved ResNet18 checkpoints and build an HTML report. Never trains."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

NAMES = ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus',
         'Smooth muscle', 'Normal colon mucosa', 'Cancer-associated stroma',
         'Colorectal adenocarcinoma epithelium']
RUNS = ['resnet18_head_20261026_221358', 'resnet18_partial_20261028_071505']


def metrics(cm):
    import numpy as np
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(9), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(9), where=support != 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(9), where=support+predicted != 0)
    return float(tp.sum()/cm.sum()), float(f1.mean()), [
        dict(class_id=i, class_name=NAMES[i], precision=float(precision[i]),
             recall=float(recall[i]), f1=float(f1[i]), support=int(support[i]))
        for i in range(9)]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a NEW output folder: {output}')
    import numpy as np
    import torch
    import torchvision
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms

    data_path = root/'A1/data/pathmnist.npz'
    if digest(data_path, 'md5') != 'a8b06965200029087d5bd730944a56c1':
        raise ValueError('Dataset MD5 mismatch')
    # Validate BOTH runs before writing any output. No pretrained download required.
    runs = []
    for name in RUNS:
        run = root/'A1/results/a1_2'/name
        config = json.loads((run/'config.json').read_text(encoding='utf-8-sig'))
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8-sig'))
        with (run/'history.csv').open(encoding='utf-8-sig', newline='') as f:
            history = list(csv.DictReader(f))
        if config['model'] != 'resnet18' or summary['status'] != 'completed':
            raise ValueError(f'Not a completed ResNet18 experiment: {run}')
        if config['purpose'] != 'full_train_validation' or len(history) != 10 or summary['epochs_completed'] != 10:
            raise ValueError(f'Expected 10 full epochs: {run}')
        if [int(row['epoch']) for row in history] != list(range(1,11)):
            raise ValueError('Unexpected epoch sequence')
        selected = max(history, key=lambda r: (float(r['val_macro_f1']), -float(r['val_loss'])))
        if int(selected['epoch']) != summary['best_epoch']:
            raise ValueError('Best epoch does not match history')
        checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['epoch'] != summary['best_epoch']:
            raise ValueError(f'Checkpoint epoch mismatch in {run.name}: '
                             f"{checkpoint['epoch']} != {summary['best_epoch']}")
        # JSON represents both Python tuples and lists as arrays. Normalize before comparing.
        saved_config = json.loads(json.dumps(checkpoint['config']))
        if saved_config != config:
            different = sorted(k for k in set(saved_config) | set(config)
                               if k not in saved_config or k not in config or saved_config[k] != config[k])
            raise ValueError(f'Checkpoint config mismatch in {run.name}; fields: {different}')
        if abs(checkpoint['validation_metrics']['macro_f1']-summary['best_val_macro_f1']) > 1e-9:
            raise ValueError('Checkpoint validation metric mismatch')
        if config['resize'] != '224x224 bilinear':
            raise ValueError('Unexpected resize; review preprocessing')
        runs.append((run, config, summary, history, checkpoint))
    with np.load(data_path, allow_pickle=False) as z:
        images = z['test_images']
        labels = z['test_labels'].reshape(-1).astype(np.int64)
    if images.shape != (7180,28,28,3) or labels.shape != (7180,) or set(np.unique(labels)) != set(range(9)):
        raise ValueError('Unexpected test split')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True)
    save_json(output/'status.json', {'status':'in_progress', 'training_performed':False})
    comparison = []
    for run, config, summary, history, checkpoint in runs:
        mode = config['mode']
        dest = output/mode
        dest.mkdir()
        transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(config['mean'], config['std'])])
        class TestData(Dataset):
            def __len__(self): return len(labels)
            def __getitem__(self, i):
                return transform(Image.fromarray(images[i]).convert('RGB')), int(labels[i])
        # Same batch size and AMP as training validation to keep evaluation consistent.
        loader = DataLoader(TestData(), batch_size=int(config['batch_size']), shuffle=False,
                            num_workers=0, pin_memory=device.type=='cuda')
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 9)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(device).eval()
        preds, probs = [], []
        total_loss = 0.0
        with torch.inference_mode():
            for step, (x,y) in enumerate(loader, 1):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type=='cuda' and bool(config['amp'])):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError('Nonfinite evaluation values')
                confidence, predicted = logits.float().softmax(1).max(1)
                preds.extend(predicted.cpu().tolist())
                probs.extend(confidence.cpu().tolist())
                total_loss += float(loss)*len(y)
                if step == 1 or step % 500 == 0:
                    print(f'{mode}: TEST {step}/{len(loader)}', flush=True)
        preds = np.asarray(preds)
        cm = np.bincount(labels*9+preds, minlength=81).reshape(9,9)
        accuracy, macro_f1, per_class = metrics(cm)
        assert int(cm.sum()) == 7180 and len(preds) == 7180
        result = dict(model='resnet18', mode=mode, split='test', samples=7180,
                      accuracy=accuracy, macro_f1=macro_f1, loss=total_loss/7180,
                      checkpoint_epoch=checkpoint['epoch'], checkpoint_sha256=digest(run/'best.pt'),
                      dataset_md5=config['dataset_md5'], source_run=run.name,
                      device=str(device), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                      amp=device.type=='cuda' and bool(config['amp']), training_performed=False,
                      selection=config['selection'], confusion_matrix_rows='true',
                      confusion_matrix_columns='predicted')
        save_json(dest/'test_metrics.json', result)
        write_csv(dest/'classification_report.csv', per_class)
        write_csv(dest/'predictions.csv', [dict(test_index=i, true_label=int(labels[i]),
                  predicted_label=int(preds[i]), confidence=float(probs[i]), correct=bool(labels[i]==preds[i]))
                  for i in range(7180)])
        np.savetxt(dest/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
        fig, ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(9), yticks=range(9), xlabel='Predicted class ID', ylabel='True class ID',
               title=f'ResNet18 {mode}: test confusion matrix')
        for i in range(9):
            for j in range(9):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(dest/'confusion_matrix.png', dpi=170); plt.close(fig)
        fig, axes = plt.subplots(2,4,figsize=(12,6),squeeze=False)
        for row, correct in enumerate([True,False]):
            indices = np.flatnonzero((preds==labels)==correct)[:4]
            for col, ax in enumerate(axes[row]):
                ax.axis('off')
                if col < len(indices):
                    i=int(indices[col]); ax.imshow(images[i], interpolation='nearest')
                    ax.set_title(f'Index {i} | True {labels[i]} / Pred {preds[i]}\nConfidence {probs[i]:.3f}',fontsize=9)
                else: ax.set_title('No example available')
        fig.suptitle('First four correct and incorrect test predictions; class IDs in class report')
        fig.tight_layout(); fig.savefig(dest/'sample_predictions.png',dpi=170); plt.close(fig)
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        epochs = [int(r['epoch']) for r in history]
        for ax, metric in zip(axes,['loss','macro_f1']):
            for split in ['train','val']:
                ax.plot(epochs,[float(r[f'{split}_{metric}']) for r in history],marker='o',label=split)
            ax.axvline(summary['best_epoch'],color='gray',linestyle='--',label='Selected epoch')
            ax.set(xlabel='Epoch',ylabel=metric,title=f'ResNet18 {mode}'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(dest/'training_curves.png',dpi=170); plt.close(fig)
        comparison.append(dict(mode=mode, epochs=10, best_epoch=summary['best_epoch'],
            val_accuracy=checkpoint['validation_metrics']['accuracy'], val_macro_f1=summary['best_val_macro_f1'],
            test_accuracy=accuracy, test_macro_f1=macro_f1,
            train_val_minutes=(summary['train_seconds']+summary['validation_seconds'])/60,
            peak_allocated_gib=summary['peak_allocated_gib'], lr=config['lr'],
            parameters_trainable=config['parameters_trainable'], checkpoint_sha256=result['checkpoint_sha256']))
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    write_csv(output/'comparison.csv', comparison)
    save_json(output/'comparison.json', comparison)
    def img(path, alt):
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(alt)}</figcaption></figure>'
    body = '<h1>A1.2 So sánh hai chế độ ResNet18 trên PathMNIST</h1><p>Bùi Tiến Đức · MSSV 2680018 · CO5085</p>'
    body += '<p>Phạm vi: head và partial, mỗi chế độ 10 epochs, seed 42. Chưa thực hiện ViT, scratch và full; chưa hoàn thành toàn bộ kế hoạch tám cấu hình. Không kết luận so sánh CNN–ViT.</p>'
    body += '<h2>Dữ liệu và phương pháp</h2><p>PathMNIST gồm 9 lớp: 89.996 train, 10.004 validation, 7.180 test. Ảnh RGB 28 × 28 được resize bilinear lên 224 × 224; chuẩn hóa ImageNet. Train có horizontal flip p=0,5; test không augmentation. AdamW, weight decay 0,01, batch size 2. Head chỉ cập nhật fc; partial cập nhật layer4 và fc. Các phần đóng băng giữ eval. Learning rate khác nhau theo bảng, nên đây là so sánh hai cấu hình, không tách riêng tác động của freeze.</p>'
    body += '<p>Checkpoint chọn bằng validation Macro-F1 cao nhất, hòa thì chọn validation loss thấp hơn. Đánh giá test dùng model.eval và inference_mode, không cập nhật trọng số. Không dùng kết quả test để điều chỉnh mô hình.</p>'
    body += '<h2>Kết quả</h2><table><tr><th>Chế độ</th><th>LR</th><th>Best epoch</th><th>Val F1</th><th>Test Acc</th><th>Test F1</th><th>Phút train + val</th><th>VRAM GiB</th></tr>'
    for r in comparison:
        body += f"<tr><td>{r['mode']}</td><td>{r['lr']}</td><td>{r['best_epoch']}</td><td>{r['val_macro_f1']:.4f}</td><td>{r['test_accuracy']:.4f}</td><td>{r['test_macro_f1']:.4f}</td><td>{r['train_val_minutes']:.2f}</td><td>{r['peak_allocated_gib']:.4f}</td></tr>"
    body += '</table><p>VRAM là đỉnh tensor do PyTorch cấp phát trong lần train, không phải tổng bộ nhớ GPU. Thời gian là tổng train và validation, không gồm toàn bộ chi phí khởi tạo/lưu file. Mỗi cấu hình chỉ có một seed; chưa có khoảng tin cậy hoặc kiểm định ý nghĩa thống kê.</p>'
    for r in comparison:
        mode=r['mode']; dest=output/mode
        body += f'<h2>ResNet18 {mode}</h2>'
        for fn, alt in [('training_curves.png','Loss và Macro-F1 theo epoch'),('confusion_matrix.png','Confusion matrix trên toàn bộ tập test'),('sample_predictions.png','Các ví dụ đúng và sai đầu tiên theo thứ tự test')]:
            body += img(dest/fn,alt)
        rows=list(csv.DictReader((dest/'classification_report.csv').open(encoding='utf-8-sig')))
        weakest=min(rows,key=lambda x:float(x['f1']))
        body += f"<p>Lớp có test F1 thấp nhất: {html.escape(weakest['class_name'])}, F1 = {float(weakest['f1']):.4f}. Cần xem ảnh để phân tích nguyên nhân; không suy diễn đặc điểm mô học chỉ từ chỉ số.</p>"
    body += '<h2>Tài liệu và tái lập</h2><p>Paper nền: DOI 10.1038/s41598-024-63094-9. Đây là phần mở rộng CNN và freeze trên PathMNIST, không phải tái lập đầy đủ paper. Dữ liệu: https://medmnist.com/. Code và metadata: https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/tree/main/A1.</p><p>SHA256 checkpoint và cấu hình đánh giá được lưu trong test_metrics.json. Checkpoint cần được công bố riêng; báo cáo này không khẳng định checkpoint đã được tải lên GitHub.</p>'
    css='body{font:16px/1.6 Arial,sans-serif;max-width:1050px;margin:35px auto;padding:0 20px;color:#182334}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccc;padding:8px}th{background:#edf2f7}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#555}h1,h2{line-height:1.25}@media print{body{margin:0;max-width:none;font-size:11pt}figure{break-inside:avoid}h2{break-after:avoid}}'
    (output/'A1_2_report.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A1.2 PathMNIST ResNet18</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    save_json(output/'status.json', {'status':'completed', 'training_performed':False,
                                   'test_samples_per_model':7180, 'modes':['head','partial']})
    print('EVALUATION COMPLETE - NO TRAINING PERFORMED', output, flush=True)


if __name__ == '__main__':
    main()
