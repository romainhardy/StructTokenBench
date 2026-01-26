"""
CLIP-style contrastive loss for aligning protein structure embeddings with functional text annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ProteinFunctionCLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss module that aligns protein structure embeddings
    with functional text annotations from PubMedBERT.
    """

    def __init__(
        self,
        structure_dim: int = 128,
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        shared_dim: int = 512,
        learnable_temperature: bool = True,
        freeze_text_encoder: bool = True,
    ):
        """
        Args:
            structure_dim: Dimension of structure encoder output (default 128)
            text_model_name: HuggingFace model name for text encoder
            shared_dim: Dimension of shared embedding space (default 512)
            learnable_temperature: Whether to learn temperature parameter
            freeze_text_encoder: Whether to freeze text encoder weights
        """
        super().__init__()

        self.shared_dim = shared_dim

        # Structure projection: Linear [structure_dim -> shared_dim] + LayerNorm
        self.structure_proj = nn.Sequential(
            nn.Linear(structure_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )

        # Load text encoder and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_dim = self.text_encoder.config.hidden_size  # 768 for PubMedBERT

        # Make text encoder parameters contiguous for DeepSpeed compatibility
        for param in self.text_encoder.parameters():
            param.data = param.data.contiguous()

        # Freeze text encoder if specified
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Text projection: Linear [768 -> shared_dim] + LayerNorm
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )

        # Learnable temperature (logit_scale initialized to ln(100) ≈ 4.6)
        # This is equivalent to temperature = 1/100 = 0.01
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(100.0)))
        else:
            self.register_buffer("logit_scale", torch.ones([]) * torch.log(torch.tensor(100.0)))

    def encode_structure(
        self,
        z: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode protein structure embeddings with masked mean pooling.

        Args:
            z: Structure encoder output [B, L, structure_dim]
            attention_mask: Boolean mask [B, L], True for valid positions

        Returns:
            Structure embeddings [B, shared_dim]
        """
        # Masked mean pooling over residues
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        z_masked = z * mask_expanded  # [B, L, structure_dim]
        sum_embeddings = z_masked.sum(dim=1)  # [B, structure_dim]
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled = sum_embeddings / sum_mask  # [B, structure_dim]

        # Project to shared dimension
        return self.structure_proj(pooled)  # [B, shared_dim]

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode text descriptions using PubMedBERT.

        Args:
            texts: List of text descriptions
            device: Device to put tensors on

        Returns:
            Text embeddings [B, shared_dim]
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get text encoder output
        with torch.no_grad() if not any(p.requires_grad for p in self.text_encoder.parameters()) else torch.enable_grad():
            outputs = self.text_encoder(**encoded)

        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Project to shared dimension
        return self.text_proj(cls_embedding)  # [B, shared_dim]

    def forward(
        self,
        z: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: list[str],
        annotation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute CLIP-style symmetric contrastive loss.

        Args:
            z: Structure encoder output [B, L, structure_dim]
            attention_mask: Boolean mask [B, L], True for valid positions
            texts: List of B text descriptions (empty string for missing)
            annotation_mask: Boolean tensor [B], True if annotation exists

        Returns:
            loss: Scalar contrastive loss
            metrics: Dictionary with accuracy and other metrics
        """
        device = z.device
        batch_size = z.size(0)

        # Count samples with annotations
        num_annotated = annotation_mask.sum().item()

        if num_annotated == 0:
            # No samples with annotations in this batch
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "contrastive_loss": torch.tensor(0.0, device=device),
                "contrastive_acc_s2t": torch.tensor(0.0, device=device),
                "contrastive_acc_t2s": torch.tensor(0.0, device=device),
                "contrastive_samples_ratio": torch.tensor(0.0, device=device),
                "logit_scale": self.logit_scale.exp().detach(),
            }

        # Filter to only annotated samples
        z_filtered = z[annotation_mask]  # [N, L, structure_dim]
        mask_filtered = attention_mask[annotation_mask]  # [N, L]
        texts_filtered = [t for t, m in zip(texts, annotation_mask.tolist()) if m]

        # Encode structure and text
        structure_embeds = self.encode_structure(z_filtered, mask_filtered)  # [N, shared_dim]
        text_embeds = self.encode_text(texts_filtered, device)  # [N, shared_dim]

        # L2 normalize embeddings
        structure_embeds = F.normalize(structure_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Compute similarity matrix with learned temperature
        logit_scale = self.logit_scale.exp()
        logits_per_structure = logit_scale * structure_embeds @ text_embeds.t()  # [N, N]
        logits_per_text = logits_per_structure.t()  # [N, N]

        # Symmetric cross-entropy loss
        n = logits_per_structure.size(0)
        labels = torch.arange(n, device=device)

        loss_s2t = F.cross_entropy(logits_per_structure, labels)
        loss_t2s = F.cross_entropy(logits_per_text, labels)
        loss = (loss_s2t + loss_t2s) / 2

        # Compute retrieval accuracies
        with torch.no_grad():
            acc_s2t = (logits_per_structure.argmax(dim=-1) == labels).float().mean()
            acc_t2s = (logits_per_text.argmax(dim=-1) == labels).float().mean()

        metrics = {
            "contrastive_loss": loss.detach(),
            "contrastive_acc_s2t": acc_s2t,
            "contrastive_acc_t2s": acc_t2s,
            "contrastive_samples_ratio": torch.tensor(num_annotated / batch_size, device=device),
            "logit_scale": logit_scale.detach(),
        }

        return loss, metrics
