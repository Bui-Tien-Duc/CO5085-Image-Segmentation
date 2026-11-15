"""CamVid UNet++ single-output adaptation. Test is not used during training."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import CamVidDataset
from models.unetpp import UNetPlusPlus
from train import run_epoch


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='A2/data/CamVid')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--base', type=int, default=16)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.size < 32:
        p.error('epochs/batch-size must be positive; size >= 32')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume and any((out / name).exists() for name in ['last.pt', 'best.pt', 'config.json']):
        p.error('Output already used. Use --resume or a new output directory.')
    torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = [CamVidDataset(a.data, s, (a.size, a.size)) for s in ['train', 'val']]
    manifest = {s: [(str(x.relative_to(Path(a.data))), str(y.relative_to(Path(a.data))))
                   for x, y in ds.samples] for s, ds in zip(['train','val'], datasets)}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if a.smoke:
        datasets = [Subset(ds, range(min(4, len(ds)))) for ds in datasets]
    loaders = [DataLoader(ds, batch_size=a.batch_size, shuffle=(i == 0), num_workers=0)
               for i, ds in enumerate(datasets)]
    config = dict(model='unetpp', base=a.base, size=a.size, batch_size=a.batch_size,
                  lr=a.lr, seed=a.seed, smoke=a.smoke, split_names_sha256=digest,
                  loss='CrossEntropy ignore_index=255', optimizer='AdamW', weight_decay=.01,
                  augmentation='none', deep_supervision=False, selection='highest val_miou; earliest tie')
    model = UNetPlusPlus(base=a.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best, best_epoch, start = [], -1., None, 1
    if a.resume:
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if ck['config'] != config:
            raise ValueError('Resume configuration or split does not match.')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        history, best, best_epoch = ck['history'], ck['best_val_miou'], ck['best_epoch']
        start = ck['epoch'] + 1
        torch.set_rng_state(ck['rng_cpu'])
        if device.type == 'cuda' and ck['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        print('RESUME from epoch', start, flush=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Device:', device, '| samples train/val:', [len(ds) for ds in datasets], flush=True)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    end = 1 if a.smoke else a.epochs
    for epoch in range(start, end + 1):
        tick = time.perf_counter()
        tr = run_epoch(model, loaders[0], criterion, optimizer, device, scaler, True)
        va = run_epoch(model, loaders[1], criterion, optimizer, device, scaler, False)
        if not all(math.isfinite(v) for v in (*tr, *va)):
            raise RuntimeError('Non-finite metrics; checkpoint not saved.')
        row = dict(epoch=epoch, train_loss=tr[0], train_miou=tr[1], train_dice=tr[2],
                   val_loss=va[0], val_miou=va[1], val_dice=va[2], seconds=time.perf_counter()-tick)
        history.append(row)
        if va[1] > best:
            best, best_epoch = va[1], epoch
            save_atomic(dict(model=model.state_dict(), config=config, epoch=epoch), out / 'best.pt')
        save_atomic(dict(model=model.state_dict(), config=config, epoch=epoch,
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), history=history,
                         best_val_miou=best, best_epoch=best_epoch, rng_cpu=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), out / 'last.pt')
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        print(f'Epoch {epoch}/{end} | val_mIoU={va[1]:.4f} | {row["seconds"]:.1f}s | LAST SAVED', flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True)['model'])
    summary = dict(status='completed', purpose='smoke_only' if a.smoke else 'full_train_validation',
                   epochs_completed=len(history), best_epoch=best_epoch, best_val_miou=best,
                   checkpoint_reload='ok', test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SMOKE CHECK: OK' if a.smoke else 'TRAINING: COMPLETE', out)
    print('Test not evaluated. Resume restores the last completed epoch; interrupted epoch is repeated.')


if __name__ == '__main__':
    main()
