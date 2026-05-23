import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model() -> nn.Module:
    """U-Net with ResNet34 encoder pretrained on ImageNet. Outputs single-channel logits."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Unet-ResNet34: {total:.2f}M params")
    return model
