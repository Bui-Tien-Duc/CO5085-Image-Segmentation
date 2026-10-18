
from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data/CamVid")

for split in ["train", "val", "test"]:
    image_dir = root / split
    mask_dir = root / f"{split}annot"

    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Masks :", len(masks))

    if not images:
        continue

    image_path = images[0]
    mask_path = mask_dir / f"{image_path.stem}_L.png"

    print("Image:", image_path.name)
    print("Mask :", mask_path.name)
    print("Mask exists:", mask_path.exists())

    if not mask_path.exists():
        continue

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    mask_array = np.array(mask)

    print("Image mode:", image.mode)
    print("Image size:", image.size)
    print("Mask mode :", mask.mode)
    print("Mask size :", mask.size)
    print("Mask shape:", mask_array.shape)

    if mask_array.ndim == 2:
        print("Unique labels:", np.unique(mask_array))
    else:
        colors = np.unique(
            mask_array.reshape(-1, mask_array.shape[-1]),
            axis=0
        )
        print("Number of colors:", len(colors))
        print("First colors:", colors[:20])

print("\nPalette file:")
print((root / "class_palette.csv").read_text()[:1500])
