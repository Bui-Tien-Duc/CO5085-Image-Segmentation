# CO5085 â€“ Semantic Image Segmentation

## 1. Giá»›i thiá»‡u

BÃ i thá»±c hÃ nh xÃ¢y dá»±ng há»‡ thá»‘ng phÃ¢n Ä‘oáº¡n ngá»¯ nghÄ©a áº£nh giao thÃ´ng Ä‘Ã´ thá»‹ báº±ng mÃ´ hÃ¬nh U-Net trÃªn bá»™ dá»¯ liá»‡u CamVid.

## 2. Bá»™ dá»¯ liá»‡u

Nguá»“n táº£i CamVid: https://github.com/stonewst/CamVid-for-Segmentation

Dá»¯ liá»‡u sá»­ dá»¥ng gá»“m 367 áº£nh train, 101 áº£nh validation vÃ  233 áº£nh test, vá»›i 11 lá»›p ngá»¯ nghÄ©a.

Sau khi táº£i, giáº£i nÃ©n dá»¯ liá»‡u vÃ o `A2/data/CamVid/`.

## 3. MÃ´i trÆ°á»ng

### Chuáº©n bá»‹ mÃ´i trÆ°á»ng

Sá»­ dá»¥ng Python 3.11. Táº¡i thÆ° má»¥c A2, táº¡o mÃ´i trÆ°á»ng:

```powershell
py -3.11 -m venv .venv
```

CÃ i cÃ¡c thÆ° viá»‡n theo phiÃªn báº£n Ä‘Ã£ ghi nháº­n:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Báº£n PyTorch trong thá»±c nghiá»‡m lÃ  2.11.0+cu128.
Náº¿u pip khÃ´ng tÃ¬m tháº¥y báº£n nÃ y trÃªn nguá»“n máº·c Ä‘á»‹nh, cÃ i PyTorch
tá»« nguá»“n wheel CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

### Chuáº©n bá»‹ dá»¯ liá»‡u

Táº£i CamVid tá»« nguá»“n Ä‘Ã£ dáº«n vÃ  sáº¯p xáº¿p nhÆ° sau:

```text
data/CamVid/
â”œâ”€â”€ class_palette.csv
â”œâ”€â”€ train/
â”œâ”€â”€ trainannot/
â”œâ”€â”€ val/
â”œâ”€â”€ valannot/
â”œâ”€â”€ test/
â””â”€â”€ testannot/
```

Mask cÃ³ tÃªn tÆ°Æ¡ng á»©ng dáº¡ng <tÃªn áº£nh>_L.png.
class_palette.csv cáº§n cÃ³ cÃ¡c cá»™t name, r, g, b.

### ÄÃ¡nh giÃ¡ checkpoint

Checkpoint Ä‘Æ°á»£c cung cáº¥p táº¡i results/best_unet.pt.
Cháº¡y táº¡i thÆ° má»¥c A2:

```powershell
.\.venv\Scripts\python.exe .\verify_checkpoint.py
```

Káº¿t quáº£ tham chiáº¿u trÃªn 233 áº£nh test:
loss khoáº£ng 0.5489, mIoU khoáº£ng 0.3806, Mean Dice khoáº£ng 0.4469.
Sai lá»‡ch nhá» cÃ³ thá»ƒ xuáº¥t hiá»‡n giá»¯a cÃ¡c mÃ´i trÆ°á»ng.

## 4. Cháº¡y chÆ°Æ¡ng trÃ¬nh
CÃ¡c lá»‡nh bÃªn dÆ°á»›i Ä‘Æ°á»£c cháº¡y táº¡i thÆ° má»¥c A2.
ÄÆ°á»ng dáº«n .venv Ã¡p dá»¥ng khi mÃ´i trÆ°á»ng áº£o náº±m trong A2.

Kiá»ƒm tra dá»¯ liá»‡u:

`python check_camvid.py`

Huáº¥n luyá»‡n:

`python train.py --epochs 10 --batch-size 2 --size 256`


ÄÃ¡nh giÃ¡ checkpoint trÃªn toÃ n bá»™ táº­p test:

```powershell
.\.venv\Scripts\python.exe .\verify_checkpoint.py
```

Táº¡o hÃ¬nh so sÃ¡nh áº£nh Ä‘áº§u vÃ o, nhÃ£n tháº­t vÃ  dá»± Ä‘oÃ¡n:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py
```

Trá»±c quan hÃ³a dá»¯ liá»‡u vÃ  biá»ƒu Ä‘á»“:

`python show_camvid.py`

`python plot_history.py`

## 5. Káº¿t quáº£ thá»±c nghiá»‡m

U-Net Ä‘Æ°á»£c huáº¥n luyá»‡n trÃªn CamVid trong 10 epochs,
batch size 2, kÃ­ch thÆ°á»›c áº£nh 256 Ã— 256.

- Optimizer: AdamW.
- Learning rate: 0.001.
- Loss: CrossEntropyLoss, ignore_index=255.
- Best validation mIoU: 0.4434 táº¡i epoch 9.
- Test mIoU: 0.3806.
- Test Mean Dice: 0.4469.
- Test loss: 0.5489.

Checkpoint Ä‘Æ°á»£c chá»n theo validation mIoU cao nháº¥t.
ÄÃ£ Ä‘Ã¡nh giÃ¡ láº¡i checkpoint trÃªn toÃ n bá»™ táº­p test;
káº¿t quáº£ chi tiáº¿t lÆ°u táº¡i results/test_metrics_verified.txt.

HÃ¬nh minh há»a:
- results/prediction.png
- results/training_curves.png

## 6. BÃ¡o cÃ¡o

BÃ¡o cÃ¡o Ä‘áº§y Ä‘á»§ Ä‘Æ°á»£c lÆ°u trong thÆ° má»¥c `reports/` cá»§a repository.

## 7. TÃ i liá»‡u tham kháº£o

- Ronneberger et al. (2015), U-Net.
- Brostow et al. (2009), CamVid.
- Zhou et al. (2020), UNet++.

## A2.1 Äá» tÃ i vÃ  káº¿ hoáº¡ch nghiÃªn cá»©u

Äá» tÃ i: So sÃ¡nh U-Net vÃ  UNet++ cho phÃ¢n Ä‘oáº¡n ngá»¯ nghÄ©a
áº£nh giao thÃ´ng Ä‘Ã´ thá»‹ trÃªn CamVid.

Paper lá»±a chá»n:
Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang,
â€œUNet++: Redesigning Skip Connections to Exploit Multiscale
Features in Image Segmentation,â€ IEEE Transactions on Medical
Imaging, vol. 39, no. 6, pp. 1856â€“1867, 2020.
DOI: 10.1109/TMI.2019.2959609.

Pháº¡m vi dá»± kiáº¿n: so sÃ¡nh hai kiáº¿n trÃºc trÃªn cÃ¹ng cÃ¡ch chia
CamVid, kÃ­ch thÆ°á»›c áº£nh vÃ  ngÃ¢n sÃ¡ch huáº¥n luyá»‡n;
Ä‘Ã¡nh giÃ¡ báº±ng mIoU vÃ  Mean Dice.

Hiá»‡n tráº¡ng: Ä‘Ã£ hoÃ n thÃ nh baseline U-Net vÃ  kiá»ƒm chá»©ng
checkpoint. UNet++ vÃ  pháº§n tá»‘i Æ°u hÃ³a chÆ°a thá»±c hiá»‡n.

Ãp dá»¥ng UNet++ trÃªn CamVid chÆ°a pháº£i tÃ¡i láº­p Ä‘áº§y Ä‘á»§
thÃ­ nghiá»‡m áº£nh y sinh trong paper gá»‘c.

## Video demo

[Video demo A2 â€“ U-Net vÃ  UNet++ trÃªn CamVid](https://github.com/Bui-Tien-Duc/CO5085-Image-Segmentation/releases/tag/Video-a2-camvid-v1)

Trong má»¥c Assets, táº£i file
`A2_Demo_UNet_UNetPP_Bui_Tien_Duc_2680018.mp4`.

Video minh há»a náº¡p checkpoint, Ä‘Ã¡nh giÃ¡ trÃªn 233 áº£nh test,
xem hÃ¬nh phÃ¢n Ä‘oáº¡n vÃ  tháº£o luáº­n káº¿t quáº£.

