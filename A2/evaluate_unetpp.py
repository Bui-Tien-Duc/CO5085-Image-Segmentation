"""Evaluate fixed best checkpoint; no training or test-based selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import CamVidDataset
from models.unetpp import UNetPlusPlus
from train import confusion_update, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='A2/data/CamVid')
    p.add_argument('--checkpoint', default='A2/results/unetpp_v1/best.pt')
    p.add_argument('--output', default='A2/results/evaluation_unetpp_v1')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = ck['config']
    if config.get('smoke'):
        raise ValueError('Refusing to evaluate a smoke checkpoint as a full experiment')
    model = UNetPlusPlus(base=config['base']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    ds = CamVidDataset(a.data, 'test', (config['size'], config['size']))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros(11, 11, dtype=torch.long, device=device)
    loss_sum, count = 0., 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    palette = np.asarray(ds.colors, dtype=np.uint8)
    with torch.no_grad():
        for step, (images, masks) in enumerate(loader, 1):
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(images); loss = criterion(logits, masks)
            confusion_update(confusion, logits, masks)
            loss_sum += loss.item() * images.size(0); count += images.size(0)
            if step == 1:
                fig, axes = plt.subplots(2, 3, figsize=(10, 6))
                for i in range(min(2, images.size(0))):
                    truth = masks[i].cpu().numpy()
                    truth_rgb = np.zeros((*truth.shape, 3), dtype=np.uint8)
                    valid = truth != 255; truth_rgb[valid] = palette[truth[valid]]
                    axes[i, 0].imshow(images[i].cpu().permute(1,2,0).numpy())
                    axes[i, 1].imshow(truth_rgb)
                    axes[i, 2].imshow(palette[logits[i].argmax(0).cpu().numpy()])
                    for j, title in enumerate(['Image', 'Ground truth (ignore=black)', 'UNet++']):
                        axes[i,j].set_title(title); axes[i,j].axis('off')
                fig.tight_layout(); fig.savefig(out / 'sample_predictions.png', dpi=150); plt.close(fig)
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f'TEST {step}/{len(loader)} | images={count}', flush=True)
    miou, dice = scores(confusion)
    cm = confusion.cpu().numpy(); tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    denom = cm.sum(0) + cm.sum(1)
    with (out / 'per_class_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class','support_pixels','iou','dice','included_in_mean'])
        for i, name in enumerate(ds.class_names):
            w.writerow([name, int(cm.sum(1)[i]), float(tp[i]/max(union[i],1)),
                        float(2*tp[i]/max(denom[i],1)), bool(union[i]>0)])
    np.savetxt(out / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    metrics = dict(status='completed', training_performed=False, model='unetpp', best_epoch=ck['epoch'],
                   test_samples=count, test_loss=loss_sum/count, test_miou=miou, test_mean_dice=dice,
                   pixel_accuracy=float(tp.sum()/max(cm.sum(),1)), ignore_index=255,
                   mean_policy='classes with union > 0; matches A2/train.py',
                   checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                   parameters=sum(p.numel() for p in model.parameters()))
    (out / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print('EVALUATION COMPLETE - NO TRAINING PERFORMED', out)


if __name__ == '__main__':
    main()
