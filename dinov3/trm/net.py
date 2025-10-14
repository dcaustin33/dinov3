import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float=2.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def _find_multiple(a, b):
    return (-(a // -b)) * b


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)

# def debug_hook(module, grad_input, grad_output):
#     print(f"\n[Hook on {module.__class__.__name__}]")
#     print(f"grad_input: {[g.shape if g is not None else None for g in grad_input]}")
#     print(f"grad_output: {[g.shape if g is not None else None for g in grad_output]}")


class TRM(torch.nn.Module):
    def __init__(
        self,
        input_cls_dim: int,
        latent_z_dim: int,
        latent_y_dim: int,
        activation: torch.nn.Module,
        hidden_dim_multiplier: float,
        num_classes: int,
        n_supervision: int = 6,
        n_latent_reasoning_steps: int = 3,
        t_recursion_steps: int = 2,
        grad_clip: float = None,
    ):
        super().__init__()
        self.input_cls_dim = input_cls_dim
        self.latent_z_dim = latent_z_dim
        self.latent_y_dim = latent_y_dim
        self.activation = activation
        self.hidden_dim_multiplier = hidden_dim_multiplier
        self.num_classes = num_classes
        self.n_supervision = n_supervision
        self.n_latent_reasoning_steps = n_latent_reasoning_steps
        self.t_recursion_steps = t_recursion_steps
        self.grad_clip = grad_clip
        
        self.in_dim = self.input_cls_dim + self.latent_z_dim + self.latent_y_dim
        self.hidden_dim = int(self.in_dim * self.hidden_dim_multiplier)
        self.out_dim = self.num_classes
        
        self.cls_head = CastedLinear(self.latent_y_dim, self.num_classes, bias=False)
        self.q_head = CastedLinear(self.latent_y_dim, 1, bias=False)
        self._initialize_net()
        
        self.embed_scale = math.sqrt(self.hidden_dim)
        embed_init_std = 1.0 / self.hidden_dim
        
        self.latent_y_embedding = nn.Parameter(trunc_normal_init_(torch.empty((1, self.latent_y_dim)), std=embed_init_std))
        self.latent_z_embedding = nn.Parameter(trunc_normal_init_(torch.empty((1, self.latent_z_dim)), std=embed_init_std))
        
    def _create_layer(self, in_dim, out_dim):
        return torch.nn.Sequential(
            CastedLinear(in_dim, out_dim, bias=False),
            self.activation(**{'hidden_size': out_dim}),
        )
        
    def _initialize_net(self):
        self.net = nn.Sequential(
            self._create_layer(self.in_dim, self.hidden_dim),
            self._create_layer(self.hidden_dim, self.in_dim),
        )
        
    def latent_recursion(self, x: torch.Tensor, y_latent: torch.Tensor, z_latent: torch.Tensor):
        x_dim = x.shape[-1]
        y_latent_dim = y_latent.shape[-1]
        z_latent_dim = z_latent.shape[-1]
        input_tensor = torch.cat([x, y_latent, z_latent], dim=-1)
        for _ in range(self.n_latent_reasoning_steps):
            output_tensor = self.net(input_tensor)
            input_tensor = output_tensor + input_tensor
        y = output_tensor[:, x_dim:x_dim+y_latent_dim]
        z = output_tensor[:, x_dim+y_latent_dim:x_dim+y_latent_dim+z_latent_dim]
        return y, z
    
    def deep_recursion(self, x: torch.Tensor, y_latent: torch.Tensor, z_latent: torch.Tensor):
        # Don't modify y_latent and z_latent in place within no_grad
        for _ in range(self.t_recursion_steps - 1):
            self.net.requires_grad_(False)
            y_latent_new, z_latent_new = self.latent_recursion(x, y_latent.detach(), z_latent.detach())
            self.net.requires_grad_(True)
            y_latent = y_latent_new
            z_latent = z_latent_new
        y_latent, z_latent = self.latent_recursion(x, y_latent, z_latent)
        cls_logits = self.cls_head(y_latent)
        q_logits = self.q_head(y_latent)
        return y_latent.detach(), z_latent.detach(), cls_logits, q_logits
        
    def forward(self, x: torch.Tensor, y_true: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None) -> dict:
        """
        Training forward pass with optimizer updates.

        Input:
            x: [bs, x_dim] - input features
            y_true: [bs] - ground truth labels
            optimizer: optimizer instance for weight updates
            scaler: GradScaler for AMP (optional)

        Returns:
            dict with keys:
                - total_loss: sum of losses across all supervision steps
                - avg_loss: average loss per supervision step
                - final_accuracy: accuracy on the last supervision step
                - avg_stopping_layer: average layer where samples stopped (via q_logits)
                - samples_per_layer: list of sample counts at each layer
        """
        batch_size = x.shape[0]
        y_latent = self.latent_y_embedding.repeat(batch_size, 1)
        z_latent = self.latent_z_embedding.repeat(batch_size, 1)

        total_loss = 0.0
        samples_per_layer = []
        final_accuracy = 0.0
        stopping_layers = torch.zeros(batch_size, device=x.device)
        stopped_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Keep original indices to track stopping layers
        original_indices = torch.arange(batch_size, device=x.device)
        weights_before = self.net[0][0].weight.clone()

        for step in range(self.n_supervision):
            current_batch_size = x.shape[0]
            samples_per_layer.append(current_batch_size)

            y_latent, z_latent, cls_logits, q_logits = self.deep_recursion(x, y_latent, z_latent)
            y_hat = torch.argmax(cls_logits, dim=-1)

            # Compute losses
            cls_loss = F.cross_entropy(cls_logits, y_true)
            q_loss = F.binary_cross_entropy_with_logits(q_logits.squeeze(-1), (y_hat == y_true).float())
            loss = cls_loss + q_loss

            # Update weights
            optimizer.zero_grad()

            if scaler is not None:
                # Use gradient scaling for AMP
                scaler.scale(loss).backward()

                # Clip gradients if specified (unscale first)
                if self.grad_clip is not None and self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward pass without AMP
                loss.backward()

                # Clip gradients if specified
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                optimizer.step()
            total_loss += loss.item()

            # Track accuracy on final layer
            if step == self.n_supervision - 1:
                final_accuracy = (y_hat == y_true).float().mean().item()

            # Filter samples based on confidence threshold
            if step < self.n_supervision - 1:
                continue_mask = q_logits.squeeze(-1) <= 0.0  # Continue if confidence is low
                stop_mask = ~continue_mask

                # Record stopping layer for samples that stop here
                if stop_mask.any():
                    stopped_indices = original_indices[stop_mask]
                    stopping_layers[stopped_indices] = step + 1
                    stopped_mask[stopped_indices] = True

                # Keep only samples that need more reasoning
                if continue_mask.any():
                    x = x[continue_mask]
                    y_true = y_true[continue_mask]
                    y_latent = y_latent[continue_mask]
                    z_latent = z_latent[continue_mask]
                    original_indices = original_indices[continue_mask]
                else:
                    break  # All samples stopped
            else:
                # Last layer - mark remaining samples
                if not stopped_mask[original_indices].all():
                    stopping_layers[original_indices[~stopped_mask[original_indices]]] = step + 1

        avg_stopping_layer = stopping_layers.float().mean().item() if batch_size > 0 else 0.0

        return {
            'total_loss': total_loss,
            'avg_loss': total_loss / self.n_supervision,
            'final_accuracy': final_accuracy,
            'avg_stopping_layer': avg_stopping_layer,
            'samples_per_layer': samples_per_layer,
        }

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Evaluation forward pass without optimizer updates. Collects detailed metrics.

        Input:
            x: [bs, x_dim] - input features
            y_true: [bs] - ground truth labels

        Returns:
            dict with keys:
                - total_loss: sum of losses across all supervision steps
                - avg_loss: average loss per supervision step
                - final_accuracy: accuracy on the last supervision step
                - layer_accuracies: list of accuracies at each supervision layer
                - layer_losses: list of losses at each supervision layer
                - avg_stopping_layer: average layer where samples stopped
                - stopping_distribution: count of samples stopping at each layer
                - samples_per_layer: list of sample counts at each layer
                - avg_confidence: average confidence (q_logits) across all layers
                - correct_confidence: average confidence for correct predictions
                - incorrect_confidence: average confidence for incorrect predictions
        """
        batch_size = x.shape[0]
        y_latent = self.latent_y_embedding.repeat(batch_size, 1)
        z_latent = self.latent_z_embedding.repeat(batch_size, 1)

        # Tracking structures
        total_loss = 0.0
        layer_accuracies = []
        layer_losses = []
        samples_per_layer = []
        stopping_layers = torch.zeros(batch_size, device=x.device)
        stopped_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        all_confidences = []
        correct_confidences = []
        incorrect_confidences = []

        # Keep original indices for tracking
        original_indices = torch.arange(batch_size, device=x.device)

        for step in range(self.n_supervision):
            current_batch_size = x.shape[0]
            samples_per_layer.append(current_batch_size)

            (y_latent, z_latent), cls_logits, q_logits = self.deep_recursion(x, y_latent, z_latent)
            y_hat = torch.argmax(cls_logits, dim=-1)

            # Compute losses and accuracy for this layer
            cls_loss = F.cross_entropy(cls_logits, y_true)
            q_loss = F.binary_cross_entropy_with_logits(q_logits.squeeze(-1), (y_hat == y_true).float())
            print("cls_loss", cls_loss.item(), "q_loss", q_loss.item())
            loss = cls_loss + q_loss

            total_loss += loss.item()
            layer_losses.append(loss.item())

            accuracy = (y_hat == y_true).float().mean().item()
            layer_accuracies.append(accuracy)

            # Track confidence statistics
            confidence = torch.sigmoid(q_logits.squeeze(-1))
            all_confidences.extend(confidence.cpu().tolist())

            correct_mask = (y_hat == y_true)
            if correct_mask.any():
                correct_confidences.extend(confidence[correct_mask].cpu().tolist())
            if (~correct_mask).any():
                incorrect_confidences.extend(confidence[~correct_mask].cpu().tolist())

            # Filter samples based on confidence threshold
            if step < self.n_supervision - 1:
                continue_mask = q_logits.squeeze(-1) <= 0.0
                stop_mask = ~continue_mask

                # Record stopping layer for samples that stop here
                if stop_mask.any():
                    stopped_indices = original_indices[stop_mask]
                    stopping_layers[stopped_indices] = step + 1
                    stopped_mask[stopped_indices] = True

                # Keep only samples that need more reasoning
                if continue_mask.any():
                    x = x[continue_mask]
                    y_true = y_true[continue_mask]
                    y_latent = y_latent[continue_mask]
                    z_latent = z_latent[continue_mask]
                    original_indices = original_indices[continue_mask]
                else:
                    break  # All samples stopped
            else:
                # Last layer - mark remaining samples
                if not stopped_mask[original_indices].all():
                    stopping_layers[original_indices[~stopped_mask[original_indices]]] = step + 1

        # Compute stopping distribution
        stopping_distribution = []
        for layer_idx in range(1, self.n_supervision + 1):
            count = (stopping_layers == layer_idx).sum().item()
            stopping_distribution.append(count)

        avg_stopping_layer = stopping_layers.float().mean().item() if batch_size > 0 else 0.0

        return {
            'total_loss': total_loss,
            'avg_loss': total_loss / self.n_supervision,
            'final_accuracy': layer_accuracies[-1] if layer_accuracies else 0.0,
            'layer_accuracies': layer_accuracies,
            'layer_losses': layer_losses,
            'avg_stopping_layer': avg_stopping_layer,
            'stopping_distribution': stopping_distribution,
            'samples_per_layer': samples_per_layer,
            'avg_confidence': sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
            'correct_confidence': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0,
            'incorrect_confidence': sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0,
        }
