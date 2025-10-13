"""
TRM (Test-Time Recursive Module) for Image Classification.

This module provides a complete training pipeline for image classification
using adaptive inference with early stopping.
"""

from dinov3.trm.model import TRMClassifier, build_model
from dinov3.trm.net import TRM
from dinov3.trm.imagenet_transform import get_eval_transforms

__all__ = [
    'TRM',
    'TRMClassifier',
    'build_model',
    'get_eval_transforms',
]
