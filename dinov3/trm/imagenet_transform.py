from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_eval_transforms(patch_size: int = 14, resolution: int | None = None):
    """
    Create ImageNet-style train/val transforms following the described evaluation protocol.

    Args:
        patch_size (int): Model patch size (typically 14 or 16).
        resolution (int | None): Custom resolution. Defaults to one yielding 1024 tokens
                                 (448 for p14, 512 for p16).
    Returns:
        (train_transform, val_transform)
    """

    # Compute resolution if not given
    default_resolution = 32 * patch_size
    side = resolution or default_resolution

    # Warn if mismatch
    if resolution and resolution != default_resolution:
        print(f"[Warning] Custom resolution {resolution} used instead of "
              f"default {default_resolution} (for 1024 tokens).")

    # --- Training Transform ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            side,
            scale=(0.95, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # --- Evaluation Transform ---
    eval_transform = transforms.Compose([
        transforms.Resize(side, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(side),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform