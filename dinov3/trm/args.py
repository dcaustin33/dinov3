import argparse
from pathlib import Path


def get_args_parser():
    """
    Create argument parser for TRM image classification training.
    """
    parser = argparse.ArgumentParser(
        description='Train TRM model for image classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ====== Data Arguments ======
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data-root', type=str, required=True,
                           help='Path to dataset root directory')
    data_group.add_argument('--dataset', type=str, default='imagenet',
                           choices=['imagenet', 'imagenet1k'],
                           help='Dataset name')
    data_group.add_argument('--num-classes', type=int, default=1000,
                           help='Number of classes in the dataset')
    data_group.add_argument('--num-workers', type=int, default=8,
                           help='Number of data loading workers')
    data_group.add_argument('--pin-memory', action='store_true', default=True,
                           help='Pin memory for data loading')
    data_group.add_argument('--images_dtype', type=str, default='float16',
                            choices=['float16', 'float32','bfloat16'],
                            help='Dtype of images')

    # ====== Transform Arguments ======
    transform_group = parser.add_argument_group('Data Transforms')
    transform_group.add_argument('--patch-size', type=int, default=14,
                                choices=[14, 16],
                                help='Patch size for the vision transformer')
    transform_group.add_argument('--resolution', type=int, default=None,
                                help='Image resolution (defaults to 32*patch_size for 1024 tokens)')

    # ====== Backbone Arguments ======
    backbone_group = parser.add_argument_group('Backbone Model')
    backbone_group.add_argument('--backbone', type=str, default='vit_small',
                               choices=['vit_small', 'vit_base', 'vit_large', 'vit_giant2'],
                               help='Vision transformer backbone architecture')
    backbone_group.add_argument('--backbone-checkpoint', type=str, default=None,
                               help='Path to pretrained backbone checkpoint')
    backbone_group.add_argument('--freeze-backbone', action='store_true',
                               help='Freeze backbone weights during training')

    # ====== TRM Model Arguments ======
    trm_group = parser.add_argument_group('TRM Model')
    trm_group.add_argument('--latent-z-dim', type=int, default=512,
                          help='Dimension of latent z variable')
    trm_group.add_argument('--latent-y-dim', type=int, default=512,
                          help='Dimension of latent y variable')
    trm_group.add_argument('--hidden-dim-multiplier', type=float, default=2.0,
                          help='Hidden dimension multiplier for TRM MLP')
    trm_group.add_argument('--n-supervision', type=int, default=4,
                          help='Number of supervision/reasoning steps')
    trm_group.add_argument('--n-latent-reasoning-steps', type=int, default=3,
                          help='Number of latent reasoning iterations per supervision step')
    trm_group.add_argument('--t-recursion-steps', type=int, default=2,
                          help='Number of deep recursion steps')
    trm_group.add_argument('--activation', type=str, default='swiglu',
                          choices=['relu', 'gelu', 'silu', 'swiglu'],
                          help='Activation function for TRM')
    trm_group.add_argument('--use-simple-mlp', action='store_true',
                          help='Use simple MLP instead of TRM for baseline comparison')
    trm_group.add_argument('--mlp-num-layers', type=int, default=2,
                          help='Number of hidden layers in MLP (only used with --use-simple-mlp)')

    # ====== Training Arguments ======
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=256,
                            help='Training batch size per GPU')
    train_group.add_argument('--eval-batch-size', type=int, default=None,
                            help='Evaluation batch size (defaults to batch_size)')
    train_group.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                            dest='learning_rate',
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.0,
                            help='Weight decay')
    train_group.add_argument('--optimizer', type=str, default='adamw',
                            choices=['adam', 'adamw', 'sgd'],
                            help='Optimizer type')
    train_group.add_argument('--momentum', type=float, default=0.9,
                            help='SGD momentum')
    train_group.add_argument('--warmup-epochs', type=int, default=5,
                            help='Number of warmup epochs')
    train_group.add_argument('--grad-clip', type=float, default=1.0,
                            help='Gradient clipping max norm (None to disable)')

    # ====== Checkpointing Arguments ======
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument('--save-dir', type=str, default='./checkpoints',
                                 help='Directory to save checkpoints and logs')
    checkpoint_group.add_argument('--experiment-name', type=str, default='trm_imagenet',
                                 help='Experiment name for logging')
    checkpoint_group.add_argument('--resume', type=str, default=None,
                                 help='Path to checkpoint to resume training from')
    checkpoint_group.add_argument('--save-freq', type=int, default=10,
                                 help='Save checkpoint every N epochs')
    checkpoint_group.add_argument('--eval-freq', type=int, default=1,
                                 help='Evaluate on validation set every N epochs')

    # ====== Logging Arguments ======
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument('--log-interval', type=int, default=50,
                              help='Log training metrics every N batches')
    logging_group.add_argument('--use-wandb', action='store_true',
                              help='Use Weights & Biases for logging')
    logging_group.add_argument('--wandb-project', type=str, default='trm-imagenet',
                              help='W&B project name')
    logging_group.add_argument('--wandb-entity', type=str, default=None,
                              help='W&B entity name')

    # ====== System Arguments ======
    system_group = parser.add_argument_group('System')
    system_group.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu', 'mps'],
                             help='Device to use for training')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    system_group.add_argument('--amp', action='store_true',
                             help='Use automatic mixed precision training')

    return parser


def postprocess_args(args):
    """
    Post-process arguments to set derived values and validate.
    """
    # Set eval batch size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Create save directory
    save_path = Path(args.save_dir) / args.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = save_path

    # Set device
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
        
    if args.images_dtype == 'float16':
        args.images_dtype = torch.float16
    elif args.images_dtype == 'float32':
        args.images_dtype = torch.float32
    elif args.images_dtype == 'bfloat16':
        args.images_dtype = torch.bfloat16

    return args


if __name__ == '__main__':
    # Test argument parser
    parser = get_args_parser()
    args = parser.parse_args(['--data-root', '/path/to/imagenet'])
    args = postprocess_args(args)
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
