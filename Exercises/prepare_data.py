"""Prepare a fixed Fashion-MNIST split for E1, E2 and E3; no training."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST


def main():
    root = Path(__file__).resolve().parent
    results = root / 'results'
    splits = root / 'splits'
    results.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    print('Downloading/checking Fashion-MNIST...', flush=True)
    train = FashionMNIST(str(root / 'data'), train=True, download=True)
    test = FashionMNIST(str(root / 'data'), train=False, download=True)
    labels = train.targets.numpy()
    test_labels = test.targets.numpy()
    rng = np.random.default_rng(42)
    train_parts, val_parts = [], []
    for class_id in range(10):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        val_parts.append(indices[:600])
        train_parts.append(indices[600:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    assert len(train_idx) == 54000 and len(val_idx) == 6000
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.unique(np.concatenate([train_idx, val_idx]))) == 60000
    split_path = splits / 'fashion_mnist_seed42.npz'
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as old:
            if not (np.array_equal(old['train_indices'], train_idx)
                    and np.array_equal(old['val_indices'], val_idx)):
                raise ValueError('Existing split differs. Preserve it and investigate before replacing.')
    else:
        np.savez_compressed(split_path, train_indices=train_idx,
                            val_indices=val_idx, test_indices=np.arange(len(test)))
    # Compute population pixel statistics from training images only, in small chunks.
    pixels = train.data.numpy()
    total = square_total = 0.0
    count = 0
    for start in range(0, len(train_idx), 1000):
        values = pixels[train_idx[start:start + 1000]].astype(np.float64) / 255.0
        total += values.sum()
        square_total += np.square(values).sum()
        count += values.size
    mean = total / count
    std = float(np.sqrt(max(0.0, square_total / count - mean * mean)))
    names = train.classes
    with (results / 'class_distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'train', 'val', 'test'])
        for c, name in enumerate(names):
            writer.writerow([c, name, int((labels[train_idx] == c).sum()),
                             int((labels[val_idx] == c).sum()), int((test_labels == c).sum())])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c, ax in enumerate(axes.flat):
        index = train_idx[np.flatnonzero(labels[train_idx] == c)[0]]
        ax.imshow(pixels[index], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{c}: {names[c]}', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(results / 'sample_images.png', dpi=180)
    plt.close(fig)
    summary = {
        'status': 'prepared', 'dataset': 'Fashion-MNIST',
        'source': 'https://github.com/zalandoresearch/fashion-mnist',
        'seed': 42, 'split_method': 'NumPy default_rng; 600 validation images per class from official training set',
        'train_samples': len(train_idx), 'val_samples': len(val_idx), 'test_samples': len(test),
        'image_shape': [1, 28, 28], 'classes': names,
        'normalization': {'mean': float(mean), 'std': std, 'computed_from': '54000 training images only; pixel / 255'},
        'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'python': sys.version, 'numpy': np.__version__,
        'training_performed': False,
    }
    (results / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('DATA PREPARATION: OK - NO TRAINING PERFORMED')
    print('Results:', results)


if __name__ == '__main__':
    main()
