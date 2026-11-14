"""Save beside train_e1.py. Evaluate four fixed best checkpoints; no training."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from train_e1 import Images, model_for

RUNS = ['softmax_v1', 'mlp_v1', 'cnn_v1', 'mlp_dropout05_v1']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(cm):
    tp = cm.diagonal().astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(10), where=precision+recall != 0)
    return precision, recall, f1, support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('E1/results/evaluation_final'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output exists. Preserve it and choose a new --output folder.')
    root = Path(__file__).resolve().parents[1]
    ex = root / 'Exercises'
    summary = read_json(ex / 'results/dataset_summary.json')
    split_hash = hashlib.sha256((ex / 'splits/fashion_mnist_seed42.npz').read_bytes()).hexdigest()
    assert split_hash == summary['split_sha256'], 'Dataset split hash mismatch'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    # Validate every run before evaluating any test data.
    prepared = []
    for run in RUNS:
        folder = root / 'E1/results' / run
        cfg, status = read_json(folder/'config.json'), read_json(folder/'summary.json')
        with (folder/'history.csv').open(encoding='utf-8-sig', newline='') as f:
            history = list(csv.DictReader(f))
        assert status['status'] == 'completed' and not cfg['smoke'], f'Incomplete run: {run}'
        assert len(history) == cfg['epochs'] == status['epochs_completed'], run
        assert [int(r['epoch']) for r in history] == list(range(1, cfg['epochs']+1)), run
        assert all(int(r['train_samples']) == 54000 and int(r['val_samples']) == 6000 for r in history), run
        assert cfg['split_sha256'] == split_hash and cfg['normalization'] == summary['normalization'], run
        best = max(history, key=lambda r: float(r['val_accuracy']))
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['config'] == cfg, f'Checkpoint configuration mismatch: {run}'
        assert checkpoint['epoch'] == int(best['epoch']), f'Wrong best epoch: {run}'
        assert abs(checkpoint['val_accuracy']-float(best['val_accuracy'])) < 1e-10, run
        assert abs(status['best_val_accuracy']-float(best['val_accuracy'])) < 1e-10, run
        net = model_for(cfg['model'], cfg['dropout'])
        assert sum(p.numel() for p in net.parameters()) == cfg['parameters'], run
        net.load_state_dict(checkpoint['model'], strict=True)
        prepared.append((run, folder, cfg, history, best, net))
    test = FashionMNIST(str(ex/'data'), train=False, download=False)
    assert len(test) == 10000
    args.output.mkdir(parents=True)
    (args.output/'status.json').write_text(json.dumps({'status':'in_progress','training_performed':False}), encoding='utf-8')
    comparisons = []
    sections = []
    for run, folder, cfg, history, best, net in prepared:
        dest = args.output/run
        dest.mkdir()
        norm = cfg['normalization']
        loader = DataLoader(Images(test, np.arange(len(test)), norm['mean'], norm['std']), batch_size=64, shuffle=False, num_workers=0)
        net = net.to(device).eval()
        predictions, labels, confidences = [], [], []
        total_loss = 0.
        with torch.inference_mode():
            for i, (x,y) in enumerate(loader):
                logits = net(x.to(device))
                total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
                confidence, pred = logits.softmax(1).max(1)
                predictions.extend(pred.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())
                labels.extend(y.tolist())
                if i % 50 == 0:
                    print(f'{run}: TEST {i+1}/{len(loader)}', flush=True)
        labels, predictions = np.asarray(labels), np.asarray(predictions)
        cm = np.bincount(labels*10+predictions, minlength=100).reshape(10,10)
        precision, recall, f1, support = scores(cm)
        checkpoint_hash = hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
        metrics = dict(run=run, model=cfg['model'], dropout=cfg['dropout'], test_samples=len(labels),
                       accuracy=float(np.trace(cm)/cm.sum()), macro_f1=float(f1.mean()), loss=total_loss/len(labels),
                       best_epoch=int(best['epoch']), best_val_accuracy=float(best['val_accuracy']),
                       parameters=cfg['parameters'], train_val_seconds=sum(float(r['train_seconds'])+float(r['val_seconds']) for r in history),
                       seed=cfg['seed'], checkpoint_sha256=checkpoint_hash, split_sha256=split_hash)
        (dest/'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        comparisons.append(metrics)
        write_csv(dest/'classification_report.csv', [dict(class_id=c, class_name=test.classes[c], precision=precision[c], recall=recall[c], f1=f1[c], support=int(support[c])) for c in range(10)])
        write_csv(dest/'predictions.csv', [dict(test_index=i, label=int(labels[i]), prediction=int(predictions[i]), confidence=confidences[i]) for i in range(len(labels))])
        write_csv(dest/'confusion_matrix.csv', [dict(true_class=c, **{f'pred_{j}':int(cm[c,j]) for j in range(10)}) for c in range(10)])
        fig, ax = plt.subplots(figsize=(8,7))
        im = ax.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted class ID', ylabel='True class ID', title=run+' - test confusion matrix')
        for c in range(10):
            for j in range(10):
                ax.text(j,c,str(cm[c,j]),ha='center',va='center',fontsize=7,color='white' if cm[c,j]>cm.max()/2 else 'black')
        fig.tight_layout(); fig.savefig(dest/'confusion_matrix.png',dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        epochs = [int(r['epoch']) for r in history]
        for ax, metric in zip(axes,['loss','accuracy']):
            for phase in ['train','val']:
                ax.plot(epochs,[float(r[f'{phase}_{metric}']) for r in history],label=phase)
            ax.set(xlabel='Epoch',ylabel=metric,title=run)
            ax.legend()
        fig.tight_layout(); fig.savefig(dest/'training_curves.png',dpi=160); plt.close(fig)
        indices = list(np.flatnonzero(labels==predictions)[:4])+list(np.flatnonzero(labels!=predictions)[:4])
        fig, axes = plt.subplots(2,4,figsize=(12,6))
        for ax in axes.flat: ax.axis('off')
        for ax, idx in zip(axes.flat,indices):
            ax.imshow(test.data[idx].numpy(),cmap='gray')
            ax.set_title(f'#{idx} True: {test.classes[labels[idx]]}\nPred: {test.classes[predictions[idx]]}\nConfidence: {confidences[idx]:.3f}',fontsize=8)
        fig.tight_layout(); fig.savefig(dest/'sample_predictions.png',dpi=160); plt.close(fig)
        sections.append(f'<h2>{html.escape(run)}</h2>'+''.join(f'<img src="{run}/{name}.png" alt="{name}" style="max-width:100%">' for name in ['training_curves','confusion_matrix','sample_predictions']))
        print(f"{run}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}",flush=True)
        net.to('cpu')
    write_csv(args.output/'comparison.csv', comparisons)
    (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    table='<table border="1"><tr><th>Run</th><th>Best epoch</th><th>Parameters</th><th>Val accuracy</th><th>Test accuracy</th><th>Test Macro-F1</th></tr>'+''.join(f"<tr><td>{r['run']}</td><td>{r['best_epoch']}</td><td>{r['parameters']}</td><td>{r['best_val_accuracy']:.2%}</td><td>{r['accuracy']:.2%}</td><td>{r['macro_f1']:.2%}</td></tr>" for r in comparisons)+'</table>'
    report='<!doctype html><html lang="vi"><meta charset="utf-8"><title>E1 Fashion-MNIST</title><body style="font-family:Arial;max-width:1100px;margin:auto"><h1>E1 - Fashion-MNIST</h1><p>Bùi Tiến Đức - 2680018. 54,000 train / 6,000 validation / 10,000 test. Seed 42; 10 epochs. Checkpoint selected by validation accuracy, earliest tie.</p>'+table+'<p>One seed only. Dropout experiment is a limited adaptation, not a complete reproduction of the original paper. Training accuracy uses train mode; validation uses eval mode. Timing includes train and validation loops, not full wall time. Test outcomes must not be used for further tuning.</p>'+''.join(sections)+'</body></html>'
    (args.output/'E1_report.html').write_text(report,encoding='utf-8')
    (args.output/'status.json').write_text(json.dumps(dict(status='completed',training_performed=False,test_samples_per_model=10000,runs=RUNS),indent=2),encoding='utf-8')
    print('EVALUATION COMPLETE - NO TRAINING PERFORMED', args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()
