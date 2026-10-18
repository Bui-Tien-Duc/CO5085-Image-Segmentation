
from pathlib import Path
import csv
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CamVidDataset(Dataset):

    NUM_CLASSES = 11
    IGNORE_INDEX = 255

    CLASS_NAMES = [
        "Sky",
        "Building",
        "Pole",
        "Road",
        "Sidewalk",
        "Tree",
        "SignSymbol",
        "Fence",
        "Car",
        "Pedestrian",
        "Bicyclist"
    ]

    def __init__(
        self,
        root="data/CamVid",
        split="train",
        size=(256, 256)
    ):

        self.root = Path(root)
        self.split = split
        self.size = size

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split: {split}"
            )

        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}annot"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image folder not found: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask folder not found: {self.mask_dir}"
            )

        # =====================================
        # 1. ĐỌC BẢNG MÀU CAMVID
        # =====================================

        palette_path = self.root / "class_palette.csv"

        if not palette_path.exists():
            raise FileNotFoundError(
                f"Palette not found: {palette_path}"
            )

        palette = {}

        with open(
            palette_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                name = row["name"]

                # Loại khoảng trắng và dấu nháy
                name = name.strip().strip("'\"").strip()

                color = (
                    int(row["r"]),
                    int(row["g"]),
                    int(row["b"])
                )

                palette[name] = color

        # =====================================
        # 2. CHỈ SỬ DỤNG 11 LỚP
        # =====================================

        self.class_names = self.CLASS_NAMES.copy()

        self.colors = []

        for name in self.class_names:

            if name not in palette:
                raise ValueError(
                    f"Class '{name}' not found in palette. "
                    f"Available classes: {list(palette.keys())}"
                )

            self.colors.append(palette[name])

        self.num_classes = len(self.class_names)

        assert self.num_classes == 11

        # =====================================
        # 3. ÁNH XẠ MÀU RGB -> CLASS ID
        # =====================================

        self.color_to_id = {
            color: class_id
            for class_id, color
            in enumerate(self.colors)
        }

        # =====================================
        # 4. TẠO DANH SÁCH ẢNH VÀ MASK
        # =====================================

        self.samples = []

        image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        for image_path in image_paths:

            mask_name = (
                image_path.stem + "_L.png"
            )

            mask_path = self.mask_dir / mask_name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            self.samples.append(
                (image_path, mask_path)
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.samples)

    # =====================================
    # 5. CHUYỂN RGB MASK -> CLASS ID
    # =====================================

    def rgb_mask_to_ids(self, mask_rgb):

        height, width, _ = mask_rgb.shape

        # Mặc định tất cả pixel là unlabeled
        mask_ids = np.full(
            (height, width),
            self.IGNORE_INDEX,
            dtype=np.int64
        )

        for color, class_id in self.color_to_id.items():

            color_array = np.array(
                color,
                dtype=np.uint8
            )

            matching_pixels = np.all(
                mask_rgb == color_array,
                axis=2
            )

            mask_ids[matching_pixels] = class_id

        return mask_ids

    # =====================================
    # 6. ĐỌC MỘT CẶP ẢNH VÀ MASK
    # =====================================

    def __getitem__(self, idx):

        image_path, mask_path = self.samples[idx]

        with Image.open(image_path) as img:

            image = img.convert("RGB")

            image = image.resize(
                self.size,
                Image.Resampling.BILINEAR
            )

        with Image.open(mask_path) as msk:

            mask = msk.convert("RGB")

            mask = mask.resize(
                self.size,
                Image.Resampling.NEAREST
            )

        # Chuẩn hóa ảnh về [0, 1]
        image_array = (
            np.asarray(
                image,
                dtype=np.float32
            ) / 255.0
        )

        # Chuyển mask màu sang class ID
        mask_rgb = np.asarray(
            mask,
            dtype=np.uint8
        )

        mask_ids = self.rgb_mask_to_ids(
            mask_rgb
        )

        # Image: [3, H, W]
        image_tensor = torch.from_numpy(
            image_array.transpose(
                2, 0, 1
            ).copy()
        ).float()

        # Mask: [H, W]
        mask_tensor = torch.from_numpy(
            mask_ids.copy()
        ).long()

        return image_tensor, mask_tensor


# =========================================
# 7. TẠO TRAIN / VAL / TEST DATALOADER
# =========================================

def create_dataloaders(
    root="data/CamVid",
    batch_size=2,
    size=(256, 256),
    num_workers=0
):

    train_dataset = CamVidDataset(
        root=root,
        split="train",
        size=size
    )

    val_dataset = CamVidDataset(
        root=root,
        split="val",
        size=size
    )

    test_dataset = CamVidDataset(
        root=root,
        split="test",
        size=size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# =========================================
# 8. KIỂM TRA DATASET
# =========================================

if __name__ == "__main__":

    print("=" * 55)
    print("CAMVID DATASET VALIDATION")
    print("=" * 55)

    for split in ["train", "val", "test"]:

        dataset = CamVidDataset(
            root="data/CamVid",
            split=split
        )

        print(f"\nSplit: {split}")

        print(
            "Number of images:",
            len(dataset)
        )

        print(
            "Number of classes:",
            dataset.num_classes
        )

        print(
            "Class names:",
            dataset.class_names
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        images, masks = next(iter(loader))

        print(
            "Images shape:",
            images.shape
        )

        print(
            "Masks shape:",
            masks.shape
        )

        print(
            "Mask IDs:",
            torch.unique(masks).tolist()
        )

        # Kiểm tra nhãn hợp lệ
        valid = (
            (
                (masks >= 0) &
                (masks < dataset.num_classes)
            )
            |
            (masks == dataset.IGNORE_INDEX)
        )

        assert bool(valid.all()), (
            "Invalid mask class IDs!"
        )

        print(
            "Dataset validation: PASSED"
        )

    print(
        "\nAll splits validated successfully!"
    )
