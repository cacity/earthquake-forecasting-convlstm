"""
Loss functions for earthquake forecasting.

This module provides various loss functions for training:
- Weighted Binary Cross-Entropy (BCE)
- Focal Loss
- Brier Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss with pos_weight parameter.

    This is the standard loss used in the paper.
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits [N, 1, H, W]
            targets: Binary targets [N, 1, H, W]

        Returns:
            Scalar loss
        """
        # BCEWithLogitsLoss with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor([self.pos_weight], device=logits.device)
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1, else 1-p

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (typically 0.25)
            gamma: Focusing parameter (typically 2.0). Higher gamma puts more
                  weight on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits [N, 1, H, W]
            targets: Binary targets [N, 1, H, W]

        Returns:
            Scalar loss
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Combine
        loss = (focal_weight * bce).mean()

        return loss


class BrierLoss(nn.Module):
    """
    Brier Score Loss (Mean Squared Error for probabilities).

    This loss directly optimizes probability calibration.

    Brier = mean((p - y)^2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits [N, 1, H, W]
            targets: Binary targets [N, 1, H, W]

        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)
        loss = F.mse_loss(probs, targets)
        return loss


class WeightedBrierLoss(nn.Module):
    """
    Weighted Brier Score Loss to handle class imbalance.

    Weighted Brier = mean(w * (p - y)^2)
    where w = pos_weight for positive class, 1 for negative class
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits [N, 1, H, W]
            targets: Binary targets [N, 1, H, W]

        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)

        # Compute squared error
        squared_error = (probs - targets).pow(2)

        # Apply weights
        weights = targets * self.pos_weight + (1 - targets)

        # Weighted mean
        loss = (weights * squared_error).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for balancing discrimination and calibration.

    Loss = lambda_bce * BCE + lambda_brier * Brier

    This can help achieve both good discrimination (ROC-AUC) and
    calibration (Brier score).
    """

    def __init__(self, pos_weight: float = 1.0,
                 lambda_bce: float = 1.0,
                 lambda_brier: float = 0.1):
        """
        Args:
            pos_weight: Weight for positive class in BCE
            lambda_bce: Weight for BCE term
            lambda_brier: Weight for Brier term
        """
        super().__init__()
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.brier_loss = BrierLoss()
        self.lambda_bce = lambda_bce
        self.lambda_brier = lambda_brier

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits [N, 1, H, W]
            targets: Binary targets [N, 1, H, W]

        Returns:
            Scalar loss
        """
        loss_bce = self.bce_loss(logits, targets)
        loss_brier = self.brier_loss(logits, targets)

        loss = self.lambda_bce * loss_bce + self.lambda_brier * loss_brier

        return loss


def get_loss_function(loss_name: str, pos_weight: float = 1.0, **kwargs) -> nn.Module:
    """
    Factory function to get loss by name.

    Args:
        loss_name: One of 'bce', 'focal', 'brier', 'weighted_brier', 'combined'
        pos_weight: Positive class weight
        **kwargs: Additional loss-specific parameters

    Returns:
        Loss function module

    Examples:
        >>> loss_fn = get_loss_function('bce', pos_weight=83.0)
        >>> loss_fn = get_loss_function('focal', alpha=0.25, gamma=2.0)
        >>> loss_fn = get_loss_function('brier')
    """
    loss_name = loss_name.lower()

    if loss_name == 'bce' or loss_name == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_name == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'brier':
        return BrierLoss()

    elif loss_name == 'weighted_brier':
        return WeightedBrierLoss(pos_weight=pos_weight)

    elif loss_name == 'combined':
        lambda_bce = kwargs.get('lambda_bce', 1.0)
        lambda_brier = kwargs.get('lambda_brier', 0.1)
        return CombinedLoss(pos_weight=pos_weight,
                           lambda_bce=lambda_bce,
                           lambda_brier=lambda_brier)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Choose from: bce, focal, brier, weighted_brier, combined")
