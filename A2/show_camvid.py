from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
root = Path("data/CamVid")
image_path = sorted((root / "val").glob("*.png"))[0]
mask_path = root / "valannot" / (image_path.stem + "_L.png")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].imshow(Image.open(image_path).convert("RGB"))
ax[0].set_title("Input image")
ax[1].imshow(Image.open(mask_path).convert("RGB"))
ax[1].set_title("Ground truth RGB mask")
for a in ax: a.axis("off")
plt.tight_layout()
Path("results").mkdir(exist_ok=True)
plt.savefig("results/camvid_sample.png", dpi=180)
plt.show()
