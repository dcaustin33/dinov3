"""
TRM Model Wrapper combining DINOv3 backbone with TRM head for image classification.
"""
import torch
import torch.nn as nn

from dinov3.models.vision_transformer import DinoVisionTransformer, vit_small, vit_base, vit_large, vit_giant2
from dinov3.trm.net import TRM, SwiGLU


class TRMClassifier(nn.Module):
    """
    Complete image classification model combining:
    - DINOv3 Vision Transformer backbone for feature extraction
    - TRM (Test-Time Recursive Module) head for adaptive inference
    """

    def __init__(
        self,
        backbone: str = 'vit_small',
        backbone_checkpoint: str = None,
        freeze_backbone: bool = True,
        patch_size: int = 16,
        num_classes: int = 1000,
        latent_z_dim: int = 512,
        latent_y_dim: int = 512,
        hidden_dim_multiplier: float = 2.0,
        n_supervision: int = 4,
        n_latent_reasoning_steps: int = 3,
        t_recursion_steps: int = 2,
        activation: str = 'relu',
        **backbone_kwargs
    ):
        """
        Args:
            backbone: Name of backbone architecture ('vit_small', 'vit_base', etc.)
            backbone_checkpoint: Path to pretrained backbone weights
            freeze_backbone: Whether to freeze backbone parameters
            patch_size: Patch size for vision transformer
            num_classes: Number of output classes
            latent_z_dim: Dimension of TRM latent z variable
            latent_y_dim: Dimension of TRM latent y variable
            hidden_dim_multiplier: Hidden dimension multiplier for TRM
            n_supervision: Number of supervision steps in TRM
            n_latent_reasoning_steps: Latent reasoning iterations per step
            t_recursion_steps: Deep recursion steps
            activation: Activation function name
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone

        # Initialize backbone
        backbone_fn_map = {
            'vit_small': vit_small,
            'vit_base': vit_base,
            'vit_large': vit_large,
            'vit_giant2': vit_giant2,
        }

        if backbone not in backbone_fn_map:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(backbone_fn_map.keys())}")

        self.backbone = backbone_fn_map[backbone](**backbone_kwargs)

        # Load pretrained weights if provided
        if backbone_checkpoint is not None:
            print(f"Loading backbone checkpoint from {backbone_checkpoint}")
            state_dict = torch.load(backbone_checkpoint, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.backbone.load_state_dict(state_dict, strict=False)

        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.embed_dim

        # Initialize activation function
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'swiglu': SwiGLU(),
        }
        activation_fn = activation_map.get(activation, SwiGLU())

        # Initialize TRM head
        self.trm = TRM(
            input_cls_dim=self.feature_dim,
            latent_z_dim=latent_z_dim,
            latent_y_dim=latent_y_dim,
            activation=activation_fn,
            hidden_dim_multiplier=hidden_dim_multiplier,
            num_classes=num_classes,
            n_supervision=n_supervision,
            n_latent_reasoning_steps=n_latent_reasoning_steps,
            t_recursion_steps=t_recursion_steps,
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using the backbone.

        Args:
            images: [B, C, H, W] input images

        Returns:
            features: [B, feature_dim] extracted features
        """
        # Handle frozen backbone
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(images, is_training=False)
        else:
            outputs = self.backbone(images, is_training=True)

        # Extract features based on configuration
        features = outputs['x_norm_clstoken']

        return features

    def forward(self, images: torch.Tensor, labels: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict:
        """
        Training forward pass.

        Args:
            images: [B, C, H, W] input images
            labels: [B] ground truth labels
            optimizer: optimizer instance for TRM weight updates

        Returns:
            dict of metrics from TRM.forward()
        """
        # Extract features from backbone
        features = self.extract_features(images)

        # Forward through TRM (with optimizer updates)
        metrics = self.trm.forward(features, labels, optimizer)

        return metrics

    def forward_eval(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Evaluation forward pass without optimizer updates.

        Args:
            images: [B, C, H, W] input images
            labels: [B] ground truth labels

        Returns:
            dict of detailed metrics from TRM.forward_eval()
        """
        # Extract features from backbone
        features = self.extract_features(images)

        # Forward through TRM (evaluation mode)
        metrics = self.trm.forward_eval(features, labels)

        return metrics

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass returning class predictions.

        Args:
            images: [B, C, H, W] input images

        Returns:
            predictions: [B] predicted class indices
        """
        self.eval()

        # Extract features
        features = self.extract_features(images)

        # Run TRM inference (simplified - just run to completion)
        batch_size = features.shape[0]
        y_latent = self.trm.latent_y_embedding.repeat(batch_size, 1)
        z_latent = self.trm.latent_z_embedding.repeat(batch_size, 1)

        # Run through all supervision steps
        for _ in range(self.trm.n_supervision):
            (y_latent, z_latent), cls_logits, _ = self.trm.deep_recursion(features, y_latent, z_latent)

        predictions = torch.argmax(cls_logits, dim=-1)
        return predictions


def build_model(args):
    """
    Build TRMClassifier from command-line arguments.

    Args:
        args: Namespace from argument parser

    Returns:
        model: TRMClassifier instance
    """
    model = TRMClassifier(
        backbone=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        freeze_backbone=args.freeze_backbone,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        latent_z_dim=args.latent_z_dim,
        latent_y_dim=args.latent_y_dim,
        hidden_dim_multiplier=args.hidden_dim_multiplier,
        n_supervision=args.n_supervision,
        n_latent_reasoning_steps=args.n_latent_reasoning_steps,
        t_recursion_steps=args.t_recursion_steps,
        activation=args.activation,
    )

    return model
