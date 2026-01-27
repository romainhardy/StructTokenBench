import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from einops import rearrange, einsum

def get_codebook_utility(input_ids, codebook_embed, eps=1e-8):
    index_count = torch.bincount(input_ids, minlength=len(codebook_embed))
    # normalize frequency to probs
    probs = index_count / torch.sum(index_count)

    # perplexity
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

    # the percentage of used indices
    num_total = len(index_count)
    use_ratio = torch.count_nonzero(index_count) / num_total

    return {
        "perplexity": perplexity,
        "perplexity_normalized": perplexity / len(codebook_embed),
        "entropy": entropy,
        "entropy_normalized": entropy / len(codebook_embed),
        "use_ratio": use_ratio,
    }


class BaseQuantizer(nn.Module):

    def __init__(self, codebook_size: int=None, codebook_embed_size: int=None, 
        loss_weight: dict=None, _need_init: bool=True, 
        freeze_codebook: bool=False, use_linear_project: bool=False, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_embed_size)

        self.loss_weight = loss_weight

        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook

        self.use_linear_project = use_linear_project
        if self.use_linear_project:
            self.linear_proj = nn.Linear(self.codebook_embed_size, self.codebook_embed_size)
    
    @torch.no_grad()
    def get_codebook(self,):
        return self.codebook.weight

    def indices2embedding(self, indices: torch.IntTensor) -> torch.Tensor:
        z_q = self.codebook[indices]
        return z_q
    
    def forward(self, z: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """Return: quantized_z, detached codes, commitment_loss
        """
        raise NotImplementedError
    
    def embedding2indices(self, z: torch.Tensor) -> torch.IntTensor:
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        
        # calculate the distance for each representation w.r.t. the codebook
        if self.use_linear_project:
            weight = self.linear_proj(self.codebook.weight)
        else:
            weight = self.codebook.weight
        dist = (torch.sum(flat_z ** 2, dim=1, keepdim=True) 
                + torch.sum(weight ** 2, dim=1) # NOTE
                - 2 * torch.matmul(flat_z, weight.t())) # [B * L, codebook_size]
        
        # get indices of the closest embedding in the codebook
        quantized_indices = torch.argmin(dist, dim=1)
        quantized_indices = rearrange(quantized_indices, "(b l) -> b l", b=batch_size, l=seq_length).detach()

        return quantized_indices

class StraightThroughQuantizer(BaseQuantizer):

    """
    Reference: https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/7a08d332f9fe9f275cdbfa82dc739fdcebad3398/vqvae/modules/vector_quantizers.py#L8
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def _tile(self, x):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L34
        """
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L43
        """
        # z: [B, L, hidden_dim]
        self._need_init = False

        flat_inputs = z.view(-1, self.codebook_embed_size) # [B * L, hidden_dim]
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]

        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.codebook.weight.detach().copy_(_k_rand)
        
        if self.freeze_codebook:
            for name, p in self.codebook.named_parameters():
                p.requires_grad = False
    
    def forward(self, z: torch.Tensor):
        # z: [B, L, hidden_dim]
        
        if self._need_init and self.training:# and not self.freeze_codebook:
            self._init_embeddings(z)
        
        # get indices of the closest embedding in the codebook
        quantized_indices = self.embedding2indices(z)

        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")
        
        quantized_z_pos = torch.zeros((flat_indices.shape[0], self.codebook_size), device=z.device)
        quantized_z_pos = quantized_z_pos.scatter_(1, flat_indices.unsqueeze(1), 1) # [B * L, codebook_size]
        if self.use_linear_project:
            quantized_z = torch.matmul(quantized_z_pos, self.linear_proj(self.codebook.weight)) # [B * L, hidden_dim = 128]
        else:
            quantized_z = torch.matmul(quantized_z_pos, self.codebook.weight) # [B * L, hidden_dim = 128]

        # loss functions
        metrics = {}
        # Reference: Eqn. (3) in https://arxiv.org/pdf/1711.00937
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss
        metrics["commitment_loss"] = commitment_loss

        quantization_loss = F.mse_loss(quantized_z, flat_z.detach())
        loss += self.loss_weight["quantization_loss_weight"] * quantization_loss
        metrics["quantization_loss"] = quantization_loss

        # straight through gradient
        quantized_z = flat_z + (quantized_z - flat_z).detach()

        quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)

        return quantized_z, quantized_indices, loss, metrics


class MCQQuantizer(nn.Module):
    """
    Multi-Codebook Quantizer (MCQ) for improved codebook utilization.

    Uses multiple independent sub-codebooks, each handling a portion of the
    embedding dimension. This improves codebook utilization and representation capacity.

    Optionally supports modality-specific codebook routing, where different modalities
    (structure, sequence, function) can be assigned to different codebooks. When a
    modality is specified during forward(), only the assigned codebooks receive gradients.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_embed_size: int,
        loss_weight: dict,
        num_codebooks: int = 8,
        use_entropy_loss: bool = False,
        entropy_temp: float = 0.01,
        use_orthogonality_loss: bool = False,  # GCP-VQVAE orthogonality regularization
        normalize_embeddings: bool = True,
        _need_init: bool = True,
        freeze_codebook: bool = False,
        # Modality-specific codebook allocation (optional)
        structure_codebooks: list = None,  # e.g., [0, 1, 2, 3] - codebooks for structure modality
        sequence_codebooks: list = None,   # e.g., [4, 5] - codebooks for sequence modality
        shared_codebooks: list = None,     # e.g., [6, 7] - codebooks shared across modalities
        **kwargs
    ):
        super().__init__()

        assert codebook_size % num_codebooks == 0, \
            f"codebook_size ({codebook_size}) must be divisible by num_codebooks ({num_codebooks})"
        assert codebook_embed_size % num_codebooks == 0, \
            f"codebook_embed_size ({codebook_embed_size}) must be divisible by num_codebooks ({num_codebooks})"

        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.num_codebooks = num_codebooks
        self.loss_weight = loss_weight

        self.sub_codebook_size = codebook_size // num_codebooks
        self.sub_embed_size = codebook_embed_size // num_codebooks

        self.use_entropy_loss = use_entropy_loss
        self.entropy_temp = entropy_temp
        self.use_orthogonality_loss = use_orthogonality_loss
        self.normalize_embeddings = normalize_embeddings

        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook

        # Modality-specific codebook allocation
        self.structure_codebooks = set(structure_codebooks) if structure_codebooks else None
        self.sequence_codebooks = set(sequence_codebooks) if sequence_codebooks else None
        self.shared_codebooks = set(shared_codebooks) if shared_codebooks else None

        # Validate codebook indices if provided
        all_indices = set(range(num_codebooks))
        for name, indices in [("structure_codebooks", self.structure_codebooks),
                              ("sequence_codebooks", self.sequence_codebooks),
                              ("shared_codebooks", self.shared_codebooks)]:
            if indices is not None:
                invalid = indices - all_indices
                assert not invalid, f"{name} contains invalid indices: {invalid}"

        self.codebooks = nn.ModuleList([
            nn.Embedding(self.sub_codebook_size, self.sub_embed_size)
            for _ in range(num_codebooks)
        ])

        self.register_buffer('vocab_usage', torch.zeros(num_codebooks, self.sub_codebook_size))

    def _get_active_codebooks_for_modality(self, modality: str = None) -> set:
        """
        Get the set of codebook indices that should receive gradients for a given modality.

        Args:
            modality: "structure", "sequence", or None (all codebooks active)

        Returns:
            Set of codebook indices that should receive gradients
        """
        if modality is None:
            return set(range(self.num_codebooks))

        active = set()

        if modality == "structure":
            if self.structure_codebooks is not None:
                active.update(self.structure_codebooks)
            else:
                # If no structure codebooks specified, use all
                return set(range(self.num_codebooks))
        elif modality == "sequence":
            if self.sequence_codebooks is not None:
                active.update(self.sequence_codebooks)
            else:
                # If no sequence codebooks specified, use all
                return set(range(self.num_codebooks))

        # Always include shared codebooks if specified
        if self.shared_codebooks is not None:
            active.update(self.shared_codebooks)

        return active if active else set(range(self.num_codebooks))

    def _get_normalized_codebook(self, codebook_idx: int) -> torch.Tensor:
        weight = self.codebooks[codebook_idx].weight
        if self.normalize_embeddings:
            return F.normalize(weight, dim=1)
        return weight

    def _tile(self, x, target_size):
        d, ew = x.shape
        if d < target_size:
            n_repeats = (target_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z: torch.Tensor):
        self._need_init = False
        chunks = z.split(self.sub_embed_size, dim=-1)

        for i, (chunk, codebook) in enumerate(zip(chunks, self.codebooks)):
            flat_chunk = chunk.reshape(-1, self.sub_embed_size)
            tiled = self._tile(flat_chunk, self.sub_codebook_size)
            init_weights = tiled[torch.randperm(tiled.shape[0])][:self.sub_codebook_size]

            if dist.is_initialized():
                dist.broadcast(init_weights, 0)
            codebook.weight.detach().copy_(init_weights)

        if self.freeze_codebook:
            for codebook in self.codebooks:
                for param in codebook.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def get_codebook(self) -> torch.Tensor:
        weights = [cb.weight for cb in self.codebooks]
        return torch.stack(weights, dim=0)

    def embedding2indices(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = z.shape
        chunks = z.split(self.sub_embed_size, dim=-1)

        all_indices = []
        for i, chunk in enumerate(chunks):
            flat_chunk = chunk.reshape(-1, self.sub_embed_size)

            if self.normalize_embeddings:
                flat_chunk = F.normalize(flat_chunk, dim=-1)

            codebook_weight = self._get_normalized_codebook(i)

            if self.normalize_embeddings:
                similarity = flat_chunk @ codebook_weight.T
                indices = torch.argmax(similarity, dim=1)
            else:
                dist_sq = (
                    flat_chunk.square().sum(dim=1, keepdim=True)
                    + codebook_weight.square().sum(dim=1)
                    - 2 * flat_chunk @ codebook_weight.T
                )
                indices = torch.argmin(dist_sq, dim=1)

            indices = indices.view(batch_size, seq_length)
            all_indices.append(indices)

        return torch.stack(all_indices, dim=1)

    def indices2embedding(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            indices = indices.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        embeddings = []
        for i in range(self.num_codebooks):
            sub_indices = indices[:, i, :]
            codebook_weight = self._get_normalized_codebook(i)
            sub_emb = F.embedding(sub_indices, codebook_weight)
            embeddings.append(sub_emb)

        result = torch.cat(embeddings, dim=-1)

        if squeeze_batch:
            result = result.squeeze(0)

        return result

    def forward(self, z: torch.Tensor, modality: str = None):
        """
        Quantize input embeddings through the multi-codebook quantizer.

        Args:
            z: [B, L, codebook_embed_size] input embeddings
            modality: Optional modality string ("structure", "sequence", or None).
                      When specified, only the codebooks assigned to that modality
                      (plus shared codebooks) receive gradients. Other codebooks
                      are used in inference mode (no gradient updates).
                      When None (default), all codebooks receive gradients.

        Returns:
            quantized_z: [B, L, codebook_embed_size] quantized embeddings
            quantized_indices: [B, num_codebooks, L] indices for each codebook
            loss: scalar loss (commitment + quantization)
            metrics: dict of metric values
        """
        if self._need_init and self.training:
            self._init_embeddings(z)

        batch_size, seq_length, _ = z.shape
        chunks = z.split(self.sub_embed_size, dim=-1)

        # Determine which codebooks should receive gradients
        active_codebooks = self._get_active_codebooks_for_modality(modality)

        all_quantized = []
        all_indices = []
        all_soft_probs = []  # For entropy loss computation
        total_commitment_loss = 0.0
        total_quantization_loss = 0.0
        total_vocab_usage = 0.0
        num_active_codebooks = 0

        for i, chunk in enumerate(chunks):
            flat_chunk = chunk.reshape(-1, self.sub_embed_size)

            if self.normalize_embeddings:
                flat_chunk_norm = F.normalize(flat_chunk, dim=-1)
            else:
                flat_chunk_norm = flat_chunk

            # Check if this codebook should receive gradients
            codebook_active = i in active_codebooks

            if codebook_active:
                # Normal path: codebook receives gradients
                codebook_weight = self._get_normalized_codebook(i)

                if self.normalize_embeddings:
                    similarity = flat_chunk_norm @ codebook_weight.T
                    indices = torch.argmax(similarity, dim=1)
                else:
                    dist_sq = (
                        flat_chunk_norm.square().sum(dim=1, keepdim=True)
                        + codebook_weight.square().sum(dim=1)
                        - 2 * flat_chunk_norm @ codebook_weight.T
                    )
                    similarity = -dist_sq  # Convert distance to similarity
                    indices = torch.argmin(dist_sq, dim=1)

                quantized = F.embedding(indices, codebook_weight)

                commitment_loss = F.mse_loss(flat_chunk_norm, quantized.detach())
                quantization_loss = F.mse_loss(quantized, flat_chunk_norm.detach())

                total_commitment_loss += commitment_loss
                total_quantization_loss += quantization_loss
                num_active_codebooks += 1

                # Compute soft assignment probabilities for entropy loss
                if self.use_entropy_loss:
                    soft_probs = F.softmax(similarity / self.entropy_temp, dim=-1)  # [B*L, sub_codebook_size]
                    all_soft_probs.append(soft_probs)

                # Straight-through estimator
                quantized = flat_chunk_norm + (quantized - flat_chunk_norm).detach()
            else:
                # Inference path: no gradients to codebook
                with torch.no_grad():
                    codebook_weight = self._get_normalized_codebook(i)

                    if self.normalize_embeddings:
                        similarity = flat_chunk_norm @ codebook_weight.T
                        indices = torch.argmax(similarity, dim=1)
                    else:
                        dist_sq = (
                            flat_chunk_norm.square().sum(dim=1, keepdim=True)
                            + codebook_weight.square().sum(dim=1)
                            - 2 * flat_chunk_norm @ codebook_weight.T
                        )
                        indices = torch.argmin(dist_sq, dim=1)

                    quantized_detached = F.embedding(indices, codebook_weight)

                # Straight-through: gradient flows to encoder but not to codebook
                quantized = flat_chunk_norm + (quantized_detached - flat_chunk_norm).detach()

            quantized = quantized.view(batch_size, seq_length, self.sub_embed_size)
            indices = indices.view(batch_size, seq_length)
            all_quantized.append(quantized)
            all_indices.append(indices)

            with torch.no_grad():
                unique_indices = torch.unique(indices)
                total_vocab_usage += len(unique_indices)

        quantized_z = torch.cat(all_quantized, dim=-1)
        quantized_indices = torch.stack(all_indices, dim=1)

        # Average losses only over active codebooks (avoid division by zero)
        if num_active_codebooks > 0:
            avg_commitment_loss = total_commitment_loss / num_active_codebooks
            avg_quantization_loss = total_quantization_loss / num_active_codebooks
        else:
            avg_commitment_loss = torch.tensor(0.0, device=z.device)
            avg_quantization_loss = torch.tensor(0.0, device=z.device)

        loss = (
            self.loss_weight["commitment_loss_weight"] * avg_commitment_loss
            + self.loss_weight["quantization_loss_weight"] * avg_quantization_loss
        )

        # Compute entropy loss to encourage uniform codebook utilization
        entropy_loss = torch.tensor(0.0, device=z.device)
        avg_entropy = torch.tensor(0.0, device=z.device)
        if self.use_entropy_loss and len(all_soft_probs) > 0:
            # Stack soft probs: [num_active_codebooks, B*L, sub_codebook_size]
            stacked_probs = torch.stack(all_soft_probs, dim=0)
            # Average across all tokens to get usage distribution per codebook
            # [num_active_codebooks, sub_codebook_size]
            avg_probs = stacked_probs.mean(dim=1)
            # Compute entropy for each codebook: H = -sum(p * log(p))
            # Higher entropy = more uniform distribution (better)
            eps = 1e-8
            entropy_per_codebook = -torch.sum(avg_probs * torch.log(avg_probs + eps), dim=-1)
            avg_entropy = entropy_per_codebook.mean()
            # Maximum possible entropy for uniform distribution
            max_entropy = torch.log(torch.tensor(self.sub_codebook_size, dtype=torch.float, device=z.device))
            # Normalize entropy to [0, 1] range and compute loss as (1 - normalized_entropy)
            # This encourages maximizing entropy (uniform usage)
            normalized_entropy = avg_entropy / max_entropy
            entropy_loss = 1.0 - normalized_entropy
            # Add weighted entropy loss
            loss = loss + self.loss_weight.get("entropy_loss_weight", 0.1) * entropy_loss

        # Compute orthogonality regularization loss (GCP-VQVAE Equation 7)
        # L_orth = ||E^T E - I_K||_F^2
        # This encourages codebook entries to be orthogonal (well-separated on hypersphere)
        orthogonality_loss = torch.tensor(0.0, device=z.device)
        if self.use_orthogonality_loss:
            total_orth_loss = 0.0
            for i in active_codebooks:
                # Get normalized codebook weights: [sub_codebook_size, sub_embed_size]
                E = self._get_normalized_codebook(i)  # [K, D] where K=64, D=16
                # Compute E^T E (Gram matrix): [K, K]
                gram = E @ E.T  # [64, 64]
                # Identity matrix
                I_K = torch.eye(self.sub_codebook_size, device=z.device, dtype=gram.dtype)
                # Frobenius norm squared: ||E^T E - I_K||_F^2
                orth_loss = torch.sum((gram - I_K) ** 2)
                total_orth_loss += orth_loss
            # Average across active codebooks
            if len(active_codebooks) > 0:
                orthogonality_loss = total_orth_loss / len(active_codebooks)
            # Add weighted orthogonality loss
            loss = loss + self.loss_weight.get("orthogonality_loss_weight", 0.1) * orthogonality_loss

        metrics = {
            "commitment_loss": avg_commitment_loss,
            "quantization_loss": avg_quantization_loss,
            "vocab_usage": total_vocab_usage / self.num_codebooks,
            "active_codebooks": num_active_codebooks,
            "entropy_loss": entropy_loss,
            "codebook_entropy": avg_entropy,
            "orthogonality_loss": orthogonality_loss,
        }

        return quantized_z, quantized_indices, loss, metrics

    def get_soft_codebook_logits(
        self,
        z: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute soft logits over codebook entries for differentiable cross-modal alignment.

        Instead of returning hard argmax indices, this returns temperature-scaled
        similarity scores (logits) that can be used for KL divergence computation
        between modalities.

        Args:
            z: [B, L, codebook_embed_size] input embeddings
            temperature: Temperature for scaling logits. Lower = sharper distribution.

        Returns:
            logits: [B, L, num_codebooks, sub_codebook_size] soft logits over codebook entries
        """
        batch_size, seq_length, _ = z.shape
        chunks = z.split(self.sub_embed_size, dim=-1)

        all_logits = []
        for i, chunk in enumerate(chunks):
            flat_chunk = chunk.reshape(-1, self.sub_embed_size)  # [B*L, sub_embed_size]

            if self.normalize_embeddings:
                flat_chunk = F.normalize(flat_chunk, dim=-1)

            codebook_weight = self._get_normalized_codebook(i)  # [sub_codebook_size, sub_embed_size]

            if self.normalize_embeddings:
                # Cosine similarity (since both are normalized)
                similarity = flat_chunk @ codebook_weight.T  # [B*L, sub_codebook_size]
            else:
                # Negative L2 distance squared (converted to similarity)
                dist_sq = (
                    flat_chunk.square().sum(dim=1, keepdim=True)
                    + codebook_weight.square().sum(dim=1)
                    - 2 * flat_chunk @ codebook_weight.T
                )
                similarity = -dist_sq  # [B*L, sub_codebook_size]

            # Scale by temperature
            logits = similarity / temperature  # [B*L, sub_codebook_size]
            logits = logits.view(batch_size, seq_length, self.sub_codebook_size)  # [B, L, sub_codebook_size]
            all_logits.append(logits)

        # Stack along codebook dimension: [B, L, num_codebooks, sub_codebook_size]
        return torch.stack(all_logits, dim=2)