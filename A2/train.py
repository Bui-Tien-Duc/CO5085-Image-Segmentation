import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from dataset import create_dataloaders
from models.unet import UNet

NUM_CLASSES = 11
IGNORE_INDEX = 255


def confusion_update(confusion, logits, targets):
    pred = logits.argmax(dim=1).reshape(-1)
    true = targets.reshape(-1)
    valid = true != IGNORE_INDEX
    idx = true[valid] * NUM_CLASSES + pred[valid]
    confusion += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def scores(confusion):
    c = confusion.float()
    tp = c.diag()
    union = c.sum(0) + c.sum(1) - tp
    denom = c.sum(0) + c.sum(1)
    iou = tp / union.clamp_min(1)
    dice = 2 * tp / denom.clamp_min(1)
    present = union > 0
    return (iou[present].mean().item() if present.any() else float('nan'),
            dice[present].mean().item() if present.any() else float('nan'))


def run_epoch(model, loader, criterion, optimizer, device, scaler, training):
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for images, masks in tqdm(loader, leave=False, desc='train' if training else 'eval'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        confusion_update(confusion, logits.detach(), masks)
    miou, mdice = scores(confusion)
    return total_loss / max(total_samples, 1), miou, mdice


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/CamVid')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--base', type=int, default=16)
    args = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data, batch_size=args.batch_size, size=(args.size, args.size), num_workers=0)
    print('Dataset sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    model = UNet(num_classes=NUM_CLASSES, base=args.base).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    results = Path('results')
    results.mkdir(exist_ok=True)
    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_loss, val_iou, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        elapsed = time.perf_counter() - start
        row = {'epoch': epoch, 'train_loss': train_loss, 'train_miou': train_iou,
               'train_dice': train_dice, 'val_loss': val_loss, 'val_miou': val_iou,
               'val_dice': val_dice, 'seconds': elapsed}
        history.append(row)
        print(f'Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} mIoU {train_iou:.4f} | '
              f'val loss {val_loss:.4f} mIoU {val_iou:.4f} Dice {val_dice:.4f} | {elapsed:.1f}s')
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({'model': model.state_dict(), 'base': args.base, 'size': args.size}, results / 'best_unet.pt')
        with (results / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
    checkpoint = torch.load(results / 'best_unet.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test_loss, test_iou, test_dice = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False)
    print(f'BEST MODEL TEST | loss={test_loss:.4f} mIoU={test_iou:.4f} Mean Dice={test_dice:.4f}')
    with (results / 'test_metrics.txt').open('w', encoding='utf-8') as f:
        f.write(f'test_loss={test_loss:.6f}\ntest_mIoU={test_iou:.6f}\ntest_mean_dice={test_dice:.6f}\n')

if __name__ == '__main__':
    main()
