from pathlib import Path

import torch
from torch import nn

from dataset import create_dataloaders
from models.unet import UNet
from train import run_epoch


def main():
    checkpoint_path = Path("results/best_unet.pt")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path.resolve()}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    _, _, test_loader = create_dataloaders(
        root="data/CamVid",
        batch_size=2,
        size=(checkpoint["size"], checkpoint["size"]),
        num_workers=0,
    )

    print("Số ảnh test:", len(test_loader.dataset))

    model = UNet(
        num_classes=11,
        base=checkpoint["base"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    loss, miou, dice = run_epoch(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        optimizer=None,
        device=device,
        scaler=None,
        training=False,
    )

    result = (
        f"test_loss={loss:.6f}\n"
        f"test_mIoU={miou:.6f}\n"
        f"test_mean_dice={dice:.6f}\n"
    )

    output = Path("results/test_metrics_verified.txt")
    output.write_text(result, encoding="utf-8")

    print("\nKẾT QUẢ ĐÁNH GIÁ CHECKPOINT")
    print(result)
    print("Đã lưu:", output.resolve())


if __name__ == "__main__":
    main()