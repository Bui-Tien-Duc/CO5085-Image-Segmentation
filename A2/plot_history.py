import csv
from pathlib import Path
import matplotlib.pyplot as plt
path = Path("results/history.csv")
with path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise ValueError("history.csv is empty")
print("CSV columns:", list(rows[0]))
def col(*names):
    for name in names:
        if name in rows[0]:
            return [float(r[name]) for r in rows]
    return None
x = col("epoch", "Epoch") or list(range(1, len(rows)+1))
series = [
    ("train_loss", "train_loss", "Train loss"),
    ("val_loss", "val_loss", "Validation loss"),
    ("val_miou", "val_miou", "Validation mIoU"),
    ("val_dice", "val_dice", "Validation Dice"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
found = False
for _, key, label in series[:2]:
    y = col(key, key.replace("_", " "))
    if y is not None:
        axes[0].plot(x, y, marker="o", label=label)
        found = True
for _, key, label in series[2:]:
    y = col(key, key.replace("_", " "))
    if y is not None:
        axes[1].plot(x, y, marker="o", label=label)
        found = True
if not found:
    raise ValueError("Update column names to match printed CSV columns")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_title("Training and validation loss")
axes[1].set_title("Validation segmentation metrics")
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=180)
plt.show()
