"""E1 Fashion-MNIST training. Save this file as E1/train_e1.py."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST


class Images(Dataset):
    def __init__(self, ds, indices, mean, std):
        self.ds, self.indices, self.mean, self.std = ds, indices, mean, std
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        j = int(self.indices[i])
        x = self.ds.data[j].unsqueeze(0).float().div(255)
        return (x - self.mean) / self.std, int(self.ds.targets[j])


def model_for(name, dropout):
    if name == 'softmax':
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if name == 'mlp':
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(128, 10))
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128, 10))


def save_pt(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['softmax', 'mlp', 'cnn'], required=True)
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 0 <= a.dropout < 1 or a.epochs < 1 or a.batch_size < 1:
        p.error('Invalid dropout, epochs or batch size')
    if a.model != 'mlp' and a.dropout:
        p.error('--dropout is only used with --model mlp')
    root = Path(__file__).resolve().parents[1]
    ex = root / 'Exercises'
    summary = json.loads((ex / 'results/dataset_summary.json').read_text(encoding='utf-8'))
    split_path = ex / 'splits/fashion_mnist_seed42.npz'
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest != summary['split_sha256']:
        raise ValueError('Split hash differs from dataset_summary.json')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = FashionMNIST(str(ex / 'data'), train=True, download=False)
    with np.load(split_path, allow_pickle=False) as split:
        ti, vi = split['train_indices'].copy(), split['val_indices'].copy()
    if a.smoke:
        ti, vi = ti[:256], vi[:128]
    norm = summary['normalization']
    generator = torch.Generator().manual_seed(42)
    train = DataLoader(Images(ds, ti, norm['mean'], norm['std']),
                       batch_size=a.batch_size, shuffle=True, generator=generator, num_workers=0)
    val = DataLoader(Images(ds, vi, norm['mean'], norm['std']),
                     batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = model_for(a.model, a.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    config = dict(model=a.model, dropout=a.dropout, epochs=a.epochs,
                  batch_size=a.batch_size, lr=a.lr, seed=42, smoke=a.smoke,
                  split_sha256=digest, normalization=norm, optimizer='Adam',
                  weight_decay=0, selection='highest validation accuracy; earliest tie',
                  parameters=sum(x.numel() for x in model.parameters()),
                  torch=str(torch.__version__), device=str(device))
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    history, best, start = [], -1., 1
    if a.resume:
        old_config = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ['model', 'dropout', 'epochs', 'batch_size', 'lr', 'seed', 'smoke', 'split_sha256', 'normalization']:
            if old_config[key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        # Only load checkpoints created locally by this script.
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best, start = state['history'], state['best'], state['epoch'] + 1
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['loader_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUME from epoch {start}', flush=True)
    else:
        if config_path.exists() or (out / 'last.pt').exists():
            raise FileExistsError('Output already contains a run. Use --resume or a new output folder.')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)
    epochs = 1 if a.smoke else a.epochs
    for epoch in range(start, epochs + 1):
        row = {'epoch': epoch}
        for phase, loader in [('train', train), ('val', val)]:
            model.train(phase == 'train')
            if device.type == 'cuda':
                torch.cuda.synchronize()
            began = time.perf_counter()
            total_loss, correct, count = 0., 0, 0
            with torch.set_grad_enabled(phase == 'train'):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss')
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item() * len(y)
                    correct += (logits.argmax(1) == y).sum().item()
                    count += len(y)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            row.update({f'{phase}_loss': total_loss/count,
                        f'{phase}_accuracy': correct/count,
                        f'{phase}_samples': count,
                        f'{phase}_seconds': time.perf_counter()-began})
        history.append(row)
        if row['val_accuracy'] > best:
            best = row['val_accuracy']
            save_pt(out / 'best.pt', {'model': model.state_dict(), 'config': config,
                                     'epoch': epoch, 'val_accuracy': best})
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     history=history, best=best, torch_rng=torch.get_rng_state(),
                     loader_rng=generator.get_state(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        save_pt(out / 'last.pt', state)
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        status = dict(status='completed' if epoch == epochs else 'in_progress',
                      purpose='smoke_check_only' if a.smoke else 'full_train_validation',
                      epochs_completed=epoch, best_val_accuracy=best,
                      test_evaluated=False)
        (out / 'summary.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(f"Epoch {epoch}/{epochs} | train_acc={row['train_accuracy']:.4f} | val_acc={row['val_accuracy']:.4f} | train_seconds={row['train_seconds']:.1f} | CHECKPOINT SAVED", flush=True)
    print('SMOKE CHECK: OK' if a.smoke else 'TRAINING: COMPLETE', out, flush=True)
    print('Test has not been evaluated. Select configurations using validation only.')


if __name__ == '__main__':
    main()
