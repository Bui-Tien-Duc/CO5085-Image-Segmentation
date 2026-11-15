"""Synthetic forward/backward and checkpoint reload; does not train CamVid."""
import io
import torch
from models.unetpp import UNetPlusPlus


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    model = UNetPlusPlus(base=16).to(device)
    image = torch.rand(2, 3, 256, 256, device=device)
    target = torch.randint(0, 11, (2, 256, 256), device=device)
    target[:, :4] = 255
    output = model(image)
    assert output.shape == (2, 11, 256, 256), output.shape
    loss = torch.nn.functional.cross_entropy(output, target, ignore_index=255)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        expected = model(image)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = UNetPlusPlus(base=16).to(device).eval()
        restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        torch.testing.assert_close(restored(image), expected)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    print('UNET++ CHECK: OK — forward, backward, checkpoint reload')
    print('Synthetic check only; no CamVid experiment completed.')


if __name__ == '__main__':
    main()
