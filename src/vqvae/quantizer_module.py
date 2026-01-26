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
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_embed_size: int,
        loss_weight: dict,
        num_codebooks: int = 8,
        use_entropy_loss: bool = False,
        entropy_temp: float = 0.01,
        normalize_embeddings: bool = True,
        _need_init: bool = True,
        freeze_codebook: bool = False,
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
        self.normalize_embeddings = normalize_embeddings

        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook

        self.codebooks = nn.ModuleList([
            nn.Embedding(self.sub_codebook_size, self.sub_embed_size)
            for _ in range(num_codebooks)
        ])

        self.register_buffer('vocab_usage', torch.zeros(num_codebooks, self.sub_codebook_size))

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

    def forward(self, z: torch.Tensor):
        if self._need_init and self.training:
            self._init_embeddings(z)

        batch_size, seq_length, _ = z.shape
        chunks = z.split(self.sub_embed_size, dim=-1)

        all_quantized = []
        all_indices = []
        total_commitment_loss = 0.0
        total_quantization_loss = 0.0
        total_vocab_usage = 0.0

        for i, chunk in enumerate(chunks):
            flat_chunk = chunk.reshape(-1, self.sub_embed_size)

            if self.normalize_embeddings:
                flat_chunk_norm = F.normalize(flat_chunk, dim=-1)
            else:
                flat_chunk_norm = flat_chunk

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

            quantized = F.embedding(indices, codebook_weight)

            commitment_loss = F.mse_loss(flat_chunk_norm, quantized.detach())
            quantization_loss = F.mse_loss(quantized, flat_chunk_norm.detach())

            total_commitment_loss += commitment_loss
            total_quantization_loss += quantization_loss

            quantized = flat_chunk_norm + (quantized - flat_chunk_norm).detach()
            quantized = quantized.view(batch_size, seq_length, self.sub_embed_size)

            indices = indices.view(batch_size, seq_length)
            all_quantized.append(quantized)
            all_indices.append(indices)

            with torch.no_grad():
                unique_indices = torch.unique(indices)
                total_vocab_usage += len(unique_indices)

        quantized_z = torch.cat(all_quantized, dim=-1)
        quantized_indices = torch.stack(all_indices, dim=1)

        avg_commitment_loss = total_commitment_loss / self.num_codebooks
        avg_quantization_loss = total_quantization_loss / self.num_codebooks

        loss = (
            self.loss_weight["commitment_loss_weight"] * avg_commitment_loss
            + self.loss_weight["quantization_loss_weight"] * avg_quantization_loss
        )

        metrics = {
            "commitment_loss": avg_commitment_loss,
            "quantization_loss": avg_quantization_loss,
            "vocab_usage": total_vocab_usage / self.num_codebooks,
        }

        return quantized_z, quantized_indices, loss, metrics