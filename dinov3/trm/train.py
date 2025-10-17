"""
Training script for TRM image classification.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from dinov3.trm.args import get_args_parser, postprocess_args
from dinov3.trm.imagenet_transform import get_eval_transforms
from dinov3.trm.model import build_model

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(args):
    """
    Create train and validation data loaders using HuggingFace datasets.

    Args:
        args: Parsed command-line arguments

    Returns:
        train_loader, val_loader
    """
    # Get transforms
    train_transform, val_transform = get_eval_transforms(
        patch_size=args.patch_size,
        resolution=args.resolution
    )

    # Load dataset using HuggingFace imagefolder
    print(f"Loading ImageNet dataset from {args.data_root}")
    dataset = load_dataset(args.data_root)

    # Split into train and validation
    train_dataset = dataset['train']
    val_dataset = dataset['validation'] if 'validation' in dataset else dataset['val']

    # Define transform functions for HuggingFace datasets
    def train_transforms(examples):
        examples['pixel_values'] = [train_transform(image.convert('RGB')) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [val_transform(image.convert('RGB')) for image in examples['image']]
        return examples

    # Apply transforms
    train_dataset = train_dataset.with_transform(train_transforms)
    val_dataset = val_dataset.with_transform(val_transforms)

    # Define collate function for HuggingFace datasets
    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([example['label'] for example in examples])
        return pixel_values, labels

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=args.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


def build_optimizer(model, args):
    """
    Build optimizer for TRM parameters.

    Note: TRM has a unique training paradigm where the optimizer is called
    inside the forward pass. We create an optimizer only for TRM parameters.

    Args:
        model: TRMClassifier instance
        args: Parsed arguments

    Returns:
        optimizer
    """
    # Only optimize TRM parameters (backbone may be frozen)
    if args.freeze_backbone:
        params = model.classifier.parameters()
        print("Optimizing only TRM parameters (backbone frozen)")
    else:
        params = model.parameters()
        print("Optimizing both backbone and TRM parameters")
        
    # print the model parameters
    for name, param in model.classifier.named_parameters():
        print(name, param.shape)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return optimizer


def get_lr_schedule(optimizer, args, steps_per_epoch):
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer instance
        args: Parsed arguments
        steps_per_epoch: Number of training steps per epoch

    Returns:
        scheduler or None
    """
    if args.warmup_epochs > 0:
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    return None


def train_epoch(model, train_loader, optimizer, scheduler, epoch, args, scaler=None, wandb_logger=None):
    """
    Train for one epoch.

    Args:
        model: TRMClassifier
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (or None)
        epoch: Current epoch number
        args: Parsed arguments
        scaler: GradScaler for AMP (or None)

    Returns:
        dict of average training metrics
    """
    model.train()
    if args.freeze_backbone:
        model.backbone.eval()  # Keep backbone in eval mode if frozen

    device = torch.device(args.device)

    total_loss = 0.0
    total_accuracy = 0.0
    total_stopping_layer = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass (TRM handles optimizer updates internally)
        if args.amp and args.device == 'cuda':
            with torch.amp.autocast(device_type=args.device, dtype=args.images_dtype):
                metrics = model.forward(images, labels, optimizer, scaler)
        else:
            metrics = model.forward(images, labels, optimizer)

        # Update learning rate if using scheduler
        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += metrics['avg_loss']
        total_accuracy += metrics['final_accuracy']
        total_stopping_layer += metrics['avg_stopping_layer']
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches
            avg_stop = total_stopping_layer / num_batches
            current_lr = optimizer.param_groups[0]['lr']

            elapsed = time.time() - start_time
            batches_per_sec = num_batches / elapsed

            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Acc: {avg_acc:.4f} "
                  f"Stop@: {avg_stop:.2f} "
                  f"LR: {current_lr:.6f} "
                  f"Speed: {batches_per_sec:.2f} batch/s")

            # Log to wandb
            if wandb_logger is not None:
                global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
                wandb_logger.log({
                    'train/batch_loss': avg_loss,
                    'train/batch_accuracy': avg_acc,
                    'train/batch_stopping_layer': avg_stop,
                    'train/learning_rate': current_lr,
                    'train/batches_per_sec': batches_per_sec,
                    'epoch': epoch,
                }, step=global_step)

    # Compute epoch averages
    avg_metrics = {
        'train/loss': total_loss / num_batches,
        'train/accuracy': total_accuracy / num_batches,
        'train/avg_stopping_layer': total_stopping_layer / num_batches,
    }

    return avg_metrics


@torch.no_grad()
def validate(model, val_loader, epoch, args, wandb_logger):
    """
    Validate on validation set.

    Args:
        model: TRMClassifier
        val_loader: Validation data loader
        epoch: Current epoch number
        args: Parsed arguments

    Returns:
        dict of validation metrics
    """
    model.eval()
    device = torch.device(args.device)

    total_loss = 0.0
    all_layer_accuracies = []
    all_stopping_layers = []
    all_stopping_distributions = []
    total_samples = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass (evaluation mode)
        metrics = model.forward_eval(images, labels)

        # Accumulate metrics
        batch_size = images.size(0)
        total_loss += metrics['avg_loss'] * batch_size
        all_layer_accuracies.append(metrics['layer_accuracies'])
        all_stopping_layers.append(metrics['avg_stopping_layer'] * batch_size)
        all_stopping_distributions.append(metrics['stopping_distribution'])
        total_samples += batch_size

        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  Validation [{batch_idx + 1}/{len(val_loader)}]")

    # Compute average metrics
    avg_loss = total_loss / total_samples
    avg_stopping_layer = sum(all_stopping_layers) / total_samples

    # Average layer accuracies across all batches
    n_layers = len(all_layer_accuracies[0])
    avg_layer_accuracies = []
    for layer_idx in range(n_layers):
        layer_acc = np.mean([batch_accs[layer_idx] for batch_accs in all_layer_accuracies])
        avg_layer_accuracies.append(layer_acc)

    # Sum stopping distributions
    stopping_distribution = np.sum(all_stopping_distributions, axis=0).tolist()

    val_metrics = {
        'val/loss': avg_loss,
        'val/accuracy': avg_layer_accuracies[-1],  # Final layer accuracy
        'val/avg_stopping_layer': avg_stopping_layer,
        'val/layer_accuracies': avg_layer_accuracies,
        'val/stopping_distribution': stopping_distribution,
    }

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Validation Results - Epoch {epoch}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {avg_layer_accuracies[-1]:.4f}")
    print(f"  Avg Stopping Layer: {avg_stopping_layer:.2f}")
    print(f"  Layer Accuracies: {[f'{acc:.3f}' for acc in avg_layer_accuracies]}")
    print(f"  Stopping Distribution: {stopping_distribution}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'='*80}\n")

    # Log to wandb
    if wandb_logger is not None:
        wandb_log = {
            'val/loss': avg_loss,
            'val/accuracy': avg_layer_accuracies[-1],
            'val/avg_stopping_layer': avg_stopping_layer,
            'val/time_seconds': elapsed,
            'epoch': epoch,
        }

        # Log per-layer accuracies
        for layer_idx, acc in enumerate(avg_layer_accuracies):
            wandb_log[f'val/layer_{layer_idx + 1}_accuracy'] = acc

        # Log stopping distribution
        for layer_idx, count in enumerate(stopping_distribution):
            wandb_log[f'val/stopping_at_layer_{layer_idx + 1}'] = count

        wandb_logger.log(wandb_log, step=epoch)

    return val_metrics


def save_checkpoint(model, optimizer, epoch, best_acc, args, is_best=False):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        best_acc: Best validation accuracy so far
        args: Parsed arguments
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'args': vars(args),
    }

    # Save latest checkpoint
    checkpoint_path = args.save_path / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = args.save_path / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")

    # Keep only latest checkpoint to save space (unless it's a save_freq milestone)
    if epoch % args.save_freq != 0 and epoch > 1:
        prev_checkpoint = args.save_path / f'checkpoint_epoch_{epoch - 1}.pth'
        if prev_checkpoint.exists() and not (epoch - 1) % args.save_freq == 0:
            prev_checkpoint.unlink()


def load_checkpoint(model, optimizer, args):
    """
    Load checkpoint to resume training.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        args: Parsed arguments

    Returns:
        start_epoch, best_acc
    """
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)

    print(f"Resumed from epoch {checkpoint['epoch']}, best acc: {best_acc:.4f}")

    return start_epoch, best_acc


def main():
    """Main training loop."""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    args = postprocess_args(args)

    # Set random seed
    set_seed(args.seed)

    # Print configuration
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("="*80 + "\n")

    # Save configuration
    config_path = args.save_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved configuration to {config_path}\n")

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("ERROR: wandb requested but not installed. Install with: pip install wandb")
            print("Continuing without wandb logging...\n")
            args.use_wandb = False
        else:
            wandb_logger = wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args),
                dir=str(args.save_path),
                resume='allow' if args.resume else None,
            )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(args)

    # Build model
    print("\nBuilding model...")
    model = build_model(args)
    device = torch.device(args.device)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Watch model with wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_logger.watch(model, log='all', log_freq=args.log_interval)
    else:
        wandb_logger = None

    # Build optimizer
    print("\nBuilding optimizer...")
    optimizer = build_optimizer(model, args)

    # Print gradient clipping info
    if args.grad_clip is not None and args.grad_clip > 0:
        print(f"Gradient clipping enabled with max norm: {args.grad_clip}")
    else:
        print("Gradient clipping disabled")

    # Build learning rate scheduler
    scheduler = get_lr_schedule(optimizer, args, len(train_loader))
    if scheduler is not None:
        print(f"Using warmup for {args.warmup_epochs} epochs + cosine decay")

    # AMP scaler
    if args.amp and args.device == 'cuda' and args.images_dtype == 'torch.float16':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args)

    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, epoch, args, scaler, wandb_logger)

        # Log epoch training metrics to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_logger.log({
                'train/epoch_loss': train_metrics['train/loss'],
                'train/epoch_accuracy': train_metrics['train/accuracy'],
                'train/epoch_stopping_layer': train_metrics['train/avg_stopping_layer'],
                'epoch': epoch,
            }, step=epoch)

        # Validate
        if epoch % args.eval_freq == 0:
            val_metrics = validate(model, val_loader, epoch, args, wandb_logger)

            # Check if best model
            current_acc = val_metrics['val/accuracy']
            is_best = current_acc > best_acc
            if is_best:
                best_acc = current_acc
                print(f"New best accuracy: {best_acc:.4f}")

            # Save checkpoint
            if epoch % args.save_freq == 0 or is_best:
                save_checkpoint(model, optimizer, epoch, best_acc, args, is_best)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print("="*80)

    # Log final summary to wandb
    if wandb_logger is not None:
        wandb_logger.run.summary['best_accuracy'] = best_acc
        wandb_logger.run.summary['final_epoch'] = args.epochs
        wandb_logger.finish()
        print(f"\nWandb run finished and logged. {wandb_logger.run.name}")


if __name__ == '__main__':
    main()
