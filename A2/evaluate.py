from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import CamVidDataset
from models.unet import UNet


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('results/best_unet.pt', map_location=device, weights_only=True)
    size = checkpoint['size']
    dataset = CamVidDataset('data/CamVid', 'test', size=(size, size))
    model = UNet(num_classes=11, base=checkpoint['base']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    image, target = dataset[0]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()
    image_np = image.permute(1, 2, 0).numpy()
    gt = target.numpy()
    colors = np.asarray(dataset.colors, dtype=np.uint8)
    def colorize(labels):
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for class_id in range(11):
            rgb[labels == class_id] = colors[class_id]
        return rgb
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [image_np, colorize(gt), colorize(prediction)],
                              ['Input', 'Ground truth', 'U-Net prediction']):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    Path('results').mkdir(exist_ok=True)
    fig.savefig('results/prediction.png', dpi=160)
    plt.close(fig)
    print('Saved results/prediction.png')

if __name__ == '__main__':
    main()
