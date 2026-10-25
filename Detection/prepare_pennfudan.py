"""Prepare Penn-Fudan YOLO labels. Run from any directory; place in Detection/."""
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw


def boxes_from_mask(mask):
    if mask.ndim != 2:
        raise ValueError('Expected a 2D instance-ID mask; do not convert to RGB.')
    boxes = []
    for ident in np.unique(mask):
        if ident == 0:
            continue
        y, x = np.where(mask == ident)
        # Pixel boundary convention: exclusive right and bottom edges.
        boxes.append((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1))
    return boxes


def main():
    base = Path(__file__).resolve().parent
    raw = base / 'data' / 'PennFudanPed'
    image_dir, mask_dir = raw / 'PNGImages', raw / 'PedMasks'
    files = sorted(image_dir.glob('*.png'))
    if len(files) != 170:
        raise RuntimeError(f'Expected 170 images, found {len(files)} at {image_dir}')
    out = base / 'data' / 'pennfudan_yolo'
    if out.exists():
        raise RuntimeError(f'Output already exists: {out}. Rename it before regenerating to prevent stale files.')
    records, hashes = [], {}
    # Validate every pair before writing anything.
    for path in files:
        mask_path = mask_dir / (path.stem + '_mask.png')
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(path) as im:
            w, h = im.size
            digest = hashlib.sha256(im.convert('RGB').tobytes() + f'{w}x{h}'.encode()).hexdigest()
        if digest in hashes:
            raise RuntimeError(f'Duplicate image pixels: {path.name}, {hashes[digest]}')
        hashes[digest] = path.name
        with Image.open(mask_path) as im:
            mask = np.asarray(im)
        if mask.shape != (h, w):
            raise ValueError(f'Mask size mismatch: {mask_path}')
        boxes = boxes_from_mask(mask)
        if not boxes:
            raise ValueError(f'No person instance: {path.name}')
        records.append((path, w, h, boxes, digest))
    random.Random(42).shuffle(records)
    splits = {'train': records[:119], 'val': records[119:144], 'test': records[144:]}
    results = base / 'results'
    results.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {}
    for split, items in splits.items():
        images = out / 'images' / split
        labels = out / 'labels' / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        counts[split] = {'images': len(items), 'boxes': 0}
        for index, (path, w, h, boxes, digest) in enumerate(items):
            shutil.copy2(path, images / path.name)
            lines = []
            for x1, y1, x2, y2 in boxes:
                xc, yc = (x1 + x2) / (2*w), (y1 + y2) / (2*h)
                bw, bh = (x2-x1)/w, (y2-y1)/h
                assert 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                lines.append(f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
            (labels / (path.stem + '.txt')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
            counts[split]['boxes'] += len(boxes)
            manifest.append({'split': split, 'image': path.name, 'boxes': len(boxes), 'pixel_sha256': digest})
            # Training and validation previews only; keep test for final evaluation.
            if split != 'test' and index < 6:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    draw = ImageDraw.Draw(im)
                    for x1, y1, x2, y2 in boxes:
                        draw.rectangle((x1, y1, x2-1, y2-1), outline='red', width=3)
                        draw.text((x1, max(0, y1-12)), 'person', fill='red')
                    im.save(results / f'label_preview_{split}_{index+1:02d}.jpg')
    yaml_path = base / 'pennfudan.yaml'
    yaml_path.write_text(f'path: {json.dumps(out.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n', encoding='utf-8')
    with (results / 'split_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split','image','boxes','pixel_sha256'])
        writer.writeheader()
        writer.writerows(manifest)
    summary = {'purpose':'dataset_preparation_only', 'seed':42, 'split_method':'random_image_split_not_official', 'counts':counts,
               'class_names':['person'], 'exact_duplicate_pixels':0,
               'near_duplicate_or_scene_leakage_review':'not_done',
               'limitations':'Random image splits do not guarantee scene or identity independence. Review train/val previews and scene groups before training; regenerate split by groups if required. No training metrics produced.'}
    (results / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('PREPARE DATASET: OK')
    print('YOLO config:', yaml_path)
    print('Review label_preview_train_*.jpg and label_preview_val_*.jpg before training.')


if __name__ == '__main__':
    main()
