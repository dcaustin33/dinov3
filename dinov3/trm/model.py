"""
TRM Model Wrapper combining DINOv3 backbone with TRM head for image classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.models.vision_transformer import DinoVisionTransformer, vit_small, vit_base, vit_large, vit_giant2
from dinov3.trm.net import TRM, SwiGLU


class SimpleMLPClassifier(nn.Module):
    """
    Simple MLP baseline classifier for benchmarking against TRM.
    Standard feedforward network without adaptive inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: nn.Module = nn.ReLU(),
        num_layers: int = 2,
        grad_clip: float = None,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            activation: Activation function
            num_layers: Number of hidden layers (minimum 1)
            grad_clip: Gradient clipping max norm (None to disable)
        """
        super().__init__()

        self.num_layers = max(1, num_layers)
        self.grad_clip = grad_clip
        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            activation(**{'hidden_size': hidden_dim}),
        ])

        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                activation(**{'hidden_size': hidden_dim}),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y_true: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict:
        """
        Training forward pass with optimizer updates (matching TRM interface).

        Args:
            x: [B, input_dim] input features
            y_true: [B] ground truth labels
            optimizer: optimizer instance for weight updates

        Returns:
            dict with metrics (matching TRM output format)
        """
        logits = self.mlp(x)
        loss = F.cross_entropy(logits, y_true)

        # Update weights
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients if specified
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == y_true).float().mean().item()

        return {
            'total_loss': loss.item(),
            'avg_loss': loss.item(),
            'final_accuracy': accuracy,
            'avg_stopping_layer': 1.0,  # Always uses single layer
            'samples_per_layer': [x.shape[0]],  # All samples in one layer
        }

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Evaluation forward pass (matching TRM interface).

        Args:
            x: [B, input_dim] input features
            y_true: [B] ground truth labels

        Returns:
            dict with detailed metrics (matching TRM output format)
        """
        logits = self.mlp(x)
        loss = F.cross_entropy(logits, y_true)

        # Compute accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == y_true).float().mean().item()

        return {
            'total_loss': loss.item(),
            'avg_loss': loss.item(),
            'final_accuracy': accuracy,
            'layer_accuracies': [accuracy],  # Single layer
            'layer_losses': [loss.item()],
            'avg_stopping_layer': 1.0,
            'stopping_distribution': [x.shape[0]],  # All samples stop at layer 1
            'samples_per_layer': [x.shape[0]],
            'avg_confidence': 1.0,  # MLP doesn't have confidence
            'correct_confidence': 1.0,
            'incorrect_confidence': 1.0,
        }


class TRMClassifier(nn.Module):
    """
    Complete image classification model combining:
    - DINOv3 Vision Transformer backbone for feature extraction
    - TRM (Test-Time Recursive Module) head for adaptive inference OR simple MLP baseline
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
        use_simple_mlp: bool = False,
        mlp_num_layers: int = 2,
        grad_clip: float = None,
        **backbone_kwargs,
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
            use_simple_mlp: If True, use simple MLP instead of TRM (for baseline comparison)
            mlp_num_layers: Number of hidden layers in MLP (only used if use_simple_mlp=True)
            grad_clip: Gradient clipping max norm (None to disable)
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.use_simple_mlp = use_simple_mlp

        # Initialize backbone
        backbone_fn_map = {
            'vit_small': vit_small,
            'vit_base': vit_base,
            'vit_large': vit_large,
            'vit_giant2': vit_giant2,
        }

        if backbone not in backbone_fn_map:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(backbone_fn_map.keys())}")

        self.backbone = backbone_fn_map[backbone](patch_size=patch_size, **backbone_kwargs)

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
            'swiglu': SwiGLU,
        }
        activation_fn = activation_map.get(activation, nn.ReLU())

        # Initialize classifier head: either TRM or simple MLP
        if use_simple_mlp:
            print(f"Using simple MLP classifier with {mlp_num_layers} hidden layers")
            # Compute hidden dim to match TRM complexity approximately
            hidden_dim = int(self.feature_dim * hidden_dim_multiplier)
            self.classifier = SimpleMLPClassifier(
                input_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                activation=activation_fn,
                num_layers=mlp_num_layers,
                grad_clip=grad_clip,
            )
        else:
            print(f"Using TRM classifier with {n_supervision} supervision steps")
            self.classifier = TRM(
                input_cls_dim=self.feature_dim,
                latent_z_dim=latent_z_dim,
                latent_y_dim=latent_y_dim,
                activation=activation_fn,
                hidden_dim_multiplier=hidden_dim_multiplier,
                num_classes=num_classes,
                n_supervision=n_supervision,
                n_latent_reasoning_steps=n_latent_reasoning_steps,
                t_recursion_steps=t_recursion_steps,
                grad_clip=grad_clip,
            )

        # Keep reference for backwards compatibility
        self.trm = self.classifier

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
            optimizer: optimizer instance for classifier weight updates

        Returns:
            dict of metrics from classifier.forward()
        """
        # Extract features from backbone
        features = self.extract_features(images)

        # Forward through classifier (TRM or MLP) with optimizer updates
        metrics = self.classifier.forward(features, labels, optimizer)

        return metrics

    def forward_eval(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Evaluation forward pass without optimizer updates.

        Args:
            images: [B, C, H, W] input images
            labels: [B] ground truth labels

        Returns:
            dict of detailed metrics from classifier.forward_eval()
        """
        # Extract features from backbone
        features = self.extract_features(images)

        # Forward through classifier (evaluation mode)
        metrics = self.classifier.forward_eval(features, labels)

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

        if self.use_simple_mlp:
            # Simple MLP prediction
            logits = self.classifier.mlp(features)
            predictions = torch.argmax(logits, dim=-1)
        else:
            # Run TRM inference (simplified - just run to completion)
            batch_size = features.shape[0]
            y_latent = self.classifier.latent_y_embedding.repeat(batch_size, 1)
            z_latent = self.classifier.latent_z_embedding.repeat(batch_size, 1)

            # Run through all supervision steps
            for _ in range(self.classifier.n_supervision):
                (y_latent, z_latent), cls_logits, _ = self.classifier.deep_recursion(features, y_latent, z_latent)

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
        use_simple_mlp=args.use_simple_mlp,
        mlp_num_layers=args.mlp_num_layers,
        grad_clip=args.grad_clip,
    )
    return model
