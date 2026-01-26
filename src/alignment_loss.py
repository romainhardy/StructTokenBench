"""
Cross-modal alignment loss for multi-stage protein representation learning.

This module implements KL divergence-based alignment between different modality
encoders (structure, sequence, function) that share a common quantizer codebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAlignmentLoss(nn.Module):
    """
    Cross-modal alignment loss using KL divergence between soft codebook assignments.

    This loss encourages different modality encoders (e.g., sequence encoder) to
    select the same codebook entries as a reference modality (e.g., structure encoder).
    It uses soft assignments (probability distributions over codebook entries) rather
    than hard indices, enabling differentiable alignment.

    The loss can be computed symmetrically (KL in both directions) or asymmetrically
    (only aligning one modality to another).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        symmetric: bool = True,
        reduction: str = "mean",
    ):
        """
        Args:
            temperature: Temperature for computing soft assignments from logits.
                        Lower values produce sharper distributions.
            symmetric: If True, compute KL divergence in both directions and average.
                      If False, only compute KL(target || source).
            reduction: How to reduce the loss ("mean", "sum", or "none").
        """
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.reduction = reduction

    def forward(
        self,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute cross-modal alignment loss between two sets of soft codebook logits.

        Args:
            source_logits: [B, L, num_codebooks, sub_codebook_size] logits from source encoder
                          (e.g., sequence encoder being trained)
            target_logits: [B, L, num_codebooks, sub_codebook_size] logits from target encoder
                          (e.g., structure encoder, typically frozen or detached)
            attention_mask: [B, L] boolean mask, True for valid positions

        Returns:
            loss: Scalar alignment loss
            metrics: Dictionary with token agreement rate and other metrics
        """
        # Convert logits to probabilities (soft assignments)
        source_probs = F.softmax(source_logits / self.temperature, dim=-1)
        target_probs = F.softmax(target_logits / self.temperature, dim=-1)

        # Compute KL divergence: KL(target || source) = sum(target * log(target/source))
        # Using log_softmax for numerical stability
        source_log_probs = F.log_softmax(source_logits / self.temperature, dim=-1)
        target_log_probs = F.log_softmax(target_logits / self.temperature, dim=-1)

        # KL(target || source) - encourages source to match target distribution
        kl_t2s = F.kl_div(source_log_probs, target_probs, reduction="none")
        # Sum over sub_codebook_size dimension: [B, L, num_codebooks]
        kl_t2s = kl_t2s.sum(dim=-1)

        if self.symmetric:
            # KL(source || target)
            kl_s2t = F.kl_div(target_log_probs, source_probs, reduction="none")
            kl_s2t = kl_s2t.sum(dim=-1)
            # Average both directions
            kl_loss = (kl_t2s + kl_s2t) / 2
        else:
            kl_loss = kl_t2s

        # Average over codebooks: [B, L]
        kl_loss = kl_loss.mean(dim=-1)

        # Apply mask if provided
        if attention_mask is not None:
            kl_loss = kl_loss * attention_mask.float()
            if self.reduction == "mean":
                num_valid = attention_mask.sum().clamp(min=1.0)
                loss = kl_loss.sum() / num_valid
            elif self.reduction == "sum":
                loss = kl_loss.sum()
            else:  # "none"
                loss = kl_loss
        else:
            if self.reduction == "mean":
                loss = kl_loss.mean()
            elif self.reduction == "sum":
                loss = kl_loss.sum()
            else:  # "none"
                loss = kl_loss

        # Compute token agreement rate (hard assignment matching)
        with torch.no_grad():
            source_indices = source_logits.argmax(dim=-1)  # [B, L, num_codebooks]
            target_indices = target_logits.argmax(dim=-1)  # [B, L, num_codebooks]
            agreement = (source_indices == target_indices).float()  # [B, L, num_codebooks]

            # Average over codebooks
            agreement = agreement.mean(dim=-1)  # [B, L]

            if attention_mask is not None:
                agreement = agreement * attention_mask.float()
                num_valid = attention_mask.sum().clamp(min=1.0)
                token_agreement_rate = agreement.sum() / num_valid
            else:
                token_agreement_rate = agreement.mean()

            # Per-codebook agreement rates
            per_codebook_agreement = (source_indices == target_indices).float()
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                per_codebook_agreement = per_codebook_agreement * mask_expanded
                per_codebook_agreement = per_codebook_agreement.sum(dim=[0, 1]) / (
                    mask_expanded.sum(dim=[0, 1]).clamp(min=1.0)
                )
            else:
                per_codebook_agreement = per_codebook_agreement.mean(dim=[0, 1])

        metrics = {
            "align_kl_loss": loss.detach() if isinstance(loss, torch.Tensor) else loss,
            "align_token_agreement_rate": token_agreement_rate,
        }

        # Add per-codebook metrics
        for i, rate in enumerate(per_codebook_agreement):
            metrics[f"align_codebook_{i}_agreement"] = rate

        return loss, metrics


class MultiModalAlignmentLoss(nn.Module):
    """
    Extended alignment loss for multiple modalities with pairwise alignment.

    Supports structure-sequence, structure-function, and sequence-function alignment
    with configurable weights for each pair.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        symmetric: bool = True,
        structure_sequence_weight: float = 1.0,
        structure_function_weight: float = 0.5,
        sequence_function_weight: float = 0.5,
    ):
        """
        Args:
            temperature: Temperature for soft assignments
            symmetric: Whether to use symmetric KL divergence
            structure_sequence_weight: Weight for structure-sequence alignment
            structure_function_weight: Weight for structure-function alignment
            sequence_function_weight: Weight for sequence-function alignment
        """
        super().__init__()
        self.base_loss = CrossModalAlignmentLoss(
            temperature=temperature,
            symmetric=symmetric,
        )
        self.structure_sequence_weight = structure_sequence_weight
        self.structure_function_weight = structure_function_weight
        self.sequence_function_weight = sequence_function_weight

    def forward(
        self,
        structure_logits: torch.Tensor | None = None,
        sequence_logits: torch.Tensor | None = None,
        function_logits: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute pairwise alignment losses between available modalities.

        Args:
            structure_logits: [B, L, num_codebooks, sub_codebook_size] or None
            sequence_logits: [B, L, num_codebooks, sub_codebook_size] or None
            function_logits: [B, L, num_codebooks, sub_codebook_size] or None
            attention_mask: [B, L] boolean mask

        Returns:
            loss: Combined alignment loss
            metrics: Dictionary with all pairwise metrics
        """
        device = None
        for logits in [structure_logits, sequence_logits, function_logits]:
            if logits is not None:
                device = logits.device
                break

        total_loss = torch.tensor(0.0, device=device)
        all_metrics = {}

        # Structure-Sequence alignment
        if structure_logits is not None and sequence_logits is not None:
            loss_ss, metrics_ss = self.base_loss(
                sequence_logits, structure_logits.detach(), attention_mask
            )
            total_loss = total_loss + self.structure_sequence_weight * loss_ss
            all_metrics.update({f"struct_seq_{k}": v for k, v in metrics_ss.items()})

        # Structure-Function alignment
        if structure_logits is not None and function_logits is not None:
            loss_sf, metrics_sf = self.base_loss(
                function_logits, structure_logits.detach(), attention_mask
            )
            total_loss = total_loss + self.structure_function_weight * loss_sf
            all_metrics.update({f"struct_func_{k}": v for k, v in metrics_sf.items()})

        # Sequence-Function alignment
        if sequence_logits is not None and function_logits is not None:
            loss_qf, metrics_qf = self.base_loss(
                function_logits, sequence_logits.detach(), attention_mask
            )
            total_loss = total_loss + self.sequence_function_weight * loss_qf
            all_metrics.update({f"seq_func_{k}": v for k, v in metrics_qf.items()})

        all_metrics["total_alignment_loss"] = total_loss.detach()

        return total_loss, all_metrics
