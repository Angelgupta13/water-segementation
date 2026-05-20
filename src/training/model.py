"""
U-Net with ResNet34 encoder pretrained on ImageNet.
Chosen for strong skip connections and proven performance on binary segmentation.
"""
import segmentation_models_pytorch as smp


def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # raw logits — BCEWithLogitsLoss handles sigmoid internally
    )
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: U-Net + ResNet34 | Total: {total:.2f}M | Trainable: {trainable:.2f}M")
    return model
