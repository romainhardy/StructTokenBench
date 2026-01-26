import os
import math
import time
import json
import hydra
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
import safetensors

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoConfig, EsmModel
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import deepspeed
from pytorch_lightning.utilities import grad_norm


from esm.layers.structure_proj import Dim6RotStructureHead
from esm.utils.constants import esm3 as C
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)
from esm.utils.structure.protein_structure import infer_cbeta_from_atom37

from modeling_util import model_init_fn
from vqvae.quantizer_module import *
from util import get_optimizer
from contrastive_loss import ProteinFunctionCLIPLoss
from annotation_loader import AnnotationLoader
from alignment_loss import CrossModalAlignmentLoss

from vqvae.blocks import VanillaUnifiedTransformerBlock
from vqvae.transformer_stack import VanillaTransformerStack
from protein_chain import WrappedProteinChain


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)

class VanillaRelativePositionEmbedding(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L20C1-L53C1
    """

    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins

        self.embedding = torch.nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        """
        Input:
          query_residue_index: (B, ) tensor of source indices (dytpe=torch.long)
          key_residue_index: (B, L) tensor of target indices (dytpe=torch.long)
        Output:
          embeddings: B x L x embedding_dim tensor of embeddings
        """

        assert query_residue_index.dtype == torch.long
        assert key_residue_index.dtype == torch.long
        assert query_residue_index.ndim == 1
        assert key_residue_index.ndim == 2

        # key_residue_index: [B * L, 16]
        # query_residue_index: [B * L, ]

        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index
        output = self.embedding(diff) # [B * L, 16, d_model=1024]
        return output

class VanillaGeometricEncoderStack(VanillaTransformerStack):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L148C1-L166C1
    """
    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList(
            [
                VanillaUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    expansion_ratio=4,
                    bias=True,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.Identity()


class VanillaStructureTokenEncoder(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L185
    """
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        # We only support fully-geometric structure token encoders for now...
        # setting n_layers_geom to something that's not n_layers won't work because
        # sequence ID isn't supported fully in this repo for plain-old transformers
        self.transformer = VanillaGeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.relative_positional_embedding = VanillaRelativePositionEmbedding(
            32, d_model, init_std=0.02
        )
        self.knn = 16
        self.d_out = d_out

    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )
            B, L, E = knn_edges.shape
            knn_edge_mask = knn_edge_mask.view(-1, E) # (B * L, 16)

            affine_tensor = affine.tensor  # for easier manipulation # [B, L, 12]
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges) # [B, L, 16, 12]
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous() # [B * L, 16, 12]
            affine = Affine3D.from_tensor(knn_affine_tensor) # [B * L, 16]
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]

            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask)

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            ) # [B * L, 16]
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask)

            knn_chain_id = torch.zeros(
                B * L, E, dtype=torch.int64, device=coords.device
            ) # [B * L, 16]

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                ) # [B * L, 16]

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs) # [B * L, 16, d_model]

        z, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        ) # [B * L, 16, d_model]

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1) # [B, L, 16, d_model]
        z = z[:, :, 0, :] # [B, L, d_model]

        return z

    @staticmethod
    def find_knn_edges(
        coords,
        padding_mask,
        coord_mask,
        sequence_id: torch.Tensor | None = None,
        knn: int | None = None,
    ) -> tuple:
        assert knn is not None, "Must specify a non-null knn to find_knn_edges"
        # Coords are N, CA, C
        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long() # [B, L]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca,
                coord_mask,
                padding_mask,
                sequence_id,
                no_knn=knn,
            )

        return edges, edge_mask # [B, L, 16], [B, L, 16]
        # edges: residue indices whose structural distance is minimized within top 16;
        # if the structural distance is masked out, use the sequence distance
        # edge_mask: True for attending, False for not

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :] # -> [B, L, 3, 3]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords) # affine: [B, L], affine_mask: [B, L]

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        ) # [B, L, d_model]

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0) # [B, L, d_model]
        z = self.pre_vq_proj(z) # [B, L, d_out]

        return z


class VanillaCategoricalMixture:
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L120C1-L146C1
    """
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(
            start, end, bins + 1, device=self.logits.device, dtype=torch.float32
        )
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ... (floating point # for target)
        true_index = (
            (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        )
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (
            self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)
        ).squeeze(-1)

    def median(self):
        return self.v_bins[self.logits.max(-1).indices]


class VanillaPairwisePredictionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L55
    """
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        n_bins: int,
        bias: bool = True,
        pairwise_state_dim: int = 0,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(
            downproject_dim + pairwise_state_dim, hidden_dim, bias=bias
        )
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [
            prod,
            diff,
        ]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class VanillaRegressionHead(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L104
    """
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x

class VanillaSequenceTokenEncoder(nn.Module):
    """
    ESM-style sequence encoder that maps amino acid tokens to latent space.
    Output can be passed through the same quantizer as structure embeddings.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_out):
        super().__init__()
        self.d_out = d_out

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer stack without geometric attention
        self.transformer = VanillaTransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=None,
            n_layers=n_layers,
            n_layers_geom=0,
            scale_residue=True,
        )

        # Project to quantizer dimension
        self.pre_vq_proj = nn.Linear(d_model, d_out)

    def encode(
        self,
        seq_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ):
        """
        Args:
            seq_tokens: [B, L] amino acid token indices
            attention_mask: [B, L] boolean, True for valid positions
            sequence_id: [B, L] optional sequence IDs

        Returns:
            z: [B, L, d_out] sequence embeddings ready for quantization
        """
        x = self.token_embedding(seq_tokens)

        if sequence_id is None:
            sequence_id = torch.zeros_like(seq_tokens, dtype=torch.int64)

        x, _ = self.transformer.forward(
            x=x,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine=None,
            affine_mask=None,
            chain_id=None,
        )

        z = self.pre_vq_proj(x)

        if attention_mask is not None:
            z = z.masked_fill(~attention_mask.unsqueeze(2), 0)

        return z


class VanillaSequenceTokenDecoder(nn.Module):
    """
    Decoder that reconstructs amino acid sequences from quantized embeddings.

    Takes the quantized latent representation and predicts per-residue amino acid
    probabilities. This enables training the sequence encoder via sequence
    reconstruction loss in addition to cross-modal alignment.
    """

    def __init__(
        self,
        encoder_d_out: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int = 33,
    ):
        """
        Args:
            encoder_d_out: Dimension of quantized embeddings (e.g., 128)
            d_model: Transformer hidden dimension (e.g., 512)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            vocab_size: Size of amino acid vocabulary (default 33 for ESM3)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Project from quantized dimension to transformer dimension
        self.post_vq_proj = nn.Linear(encoder_d_out, d_model)

        # Transformer stack for sequence modeling
        self.decoder_stack = VanillaTransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=None,
            n_layers=n_layers,
            n_layers_geom=0,
            scale_residue=False,
        )

        # Output projection to vocabulary
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def decode(
        self,
        quantized_z: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> dict:
        """
        Decode quantized embeddings to amino acid logits.

        Args:
            quantized_z: [B, L, encoder_d_out] quantized embeddings
            attention_mask: [B, L] boolean mask, True for valid positions
            sequence_id: [B, L] optional sequence IDs

        Returns:
            Dictionary with:
                - logits: [B, L, vocab_size] amino acid logits
                - last_hidden_state: [B, L, d_model] transformer output
        """
        batch_size, seq_length, _ = quantized_z.shape

        if sequence_id is None:
            sequence_id = torch.zeros(
                batch_size, seq_length, dtype=torch.int64, device=quantized_z.device
            )

        # Project to transformer dimension
        x = self.post_vq_proj(quantized_z)  # [B, L, d_model]

        # Pass through transformer
        chain_id = torch.zeros_like(sequence_id)
        x, _ = self.decoder_stack.forward(
            x=x,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine=None,
            affine_mask=None,
            chain_id=chain_id,
        )  # [B, L, d_model]

        # Project to vocabulary
        logits = self.output_head(x)  # [B, L, vocab_size]

        return {
            "logits": logits,
            "last_hidden_state": x,
        }


class VanillaStructureTokenDecoder(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L335
    """
    def __init__(
        self,
        encoder_d_out,
        d_model,
        n_heads,
        n_layers,
    ):
        super().__init__()
        self.decoder_channels = d_model

        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN

        self.post_vq_proj = nn.Linear(encoder_d_out, d_model)
        self.decoder_stack = VanillaTransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0
        )

        self.affine_output_projection = Dim6RotStructureHead(
            self.decoder_channels, 10, predict_torsion_angles=False
        )

        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [
            64,  # distogram
            direction_loss_bins * 6,  # direction bins
            pae_bins,  # predicted aligned error
        ]
        self.pairwise_classification_head = VanillaPairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = VanillaRegressionHead(
            embed_dim=self.decoder_channels, output_dim=plddt_bins
        )

    def decode(
        self,
        quantized_z: torch.Tensor,
        structure_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        skip_pairwise: bool = False,
    ):
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        x = self.post_vq_proj(quantized_z) # [B, L, hidden_dim=128] -> [B, L, d_model=1024]
        # !!! NOTE: Attention mask is actually unused here so watch out
        x, _ = self.decoder_stack.forward(
            x, attention_mask=attention_mask, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
        ) # [B, L, d_model], [B, L, d_model]

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        ) # [B, L, 12], [B, L, 3, 3]

        pae, ptm = None, None
        pairwise_dist_logits, pairwise_dir_logits = None, None

        if not skip_pairwise:
            pairwise_logits = self.pairwise_classification_head(x) # [B, L, L, 64 + 96 + 64]
            pairwise_dist_logits, pairwise_dir_logits, pae_logits = [
                (o if o.numel() > 0 else None)
                for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
            ] # [B, L, L, 64], [B, L, L, 96], [B, L, L, 64]

            special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
            pae = compute_predicted_aligned_error(
                pae_logits,  # type: ignore
                aa_mask=~special_tokens_mask,
                sequence_id=sequence_id,
                max_bin=self.max_pae_bin,
            ) # [B, L, L]
            # This might be broken for chainbreak tokens? We might align to the chainbreak
            ptm = compute_tm(
                pae_logits,  # type: ignore
                aa_mask=~special_tokens_mask,
                max_bin=self.max_pae_bin,
            ) # [B,]

        plddt_logits = self.plddt_head(x) # [B, L, 50]
        plddt_value = VanillaCategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean() # [B, L]

        return dict(
            tensor7_affine=tensor7_affine,
            bb_pred=bb_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
            pairwise_dist_logits=pairwise_dist_logits,
            pairwise_dir_logits=pairwise_dir_logits,
            last_hidden_state=x,
        )


class VQVAEModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        quantizer_cfg = model_cfg.quantizer
        self.loss_weight = quantizer_cfg["loss_weight"]
        self.quantizer = eval(quantizer_cfg["quantizer_type"])(**quantizer_cfg)

        self.encoder = VanillaStructureTokenEncoder(
            **model_cfg.encoder,
            n_codes=quantizer_cfg.codebook_size
        ) # encoder_d_out not necessarily the same as self.codebook_embed_size
        model_cfg.decoder["encoder_d_out"] = model_cfg.encoder.d_out
        self.decoder = VanillaStructureTokenDecoder(**model_cfg.decoder)

        self.inverse_folding_head = VanillaRegressionHead(
            embed_dim=model_cfg.decoder.d_model,
            output_dim=len(C.SEQUENCE_VOCAB)
        )

        self._step_count = 0

        # Multi-stage training support
        self.training_stage = model_cfg.get("training_stage", None)

        # Initialize contrastive loss if configured
        contrastive_cfg = model_cfg.get("contrastive", None)
        self.use_contrastive = contrastive_cfg is not None and contrastive_cfg.get("enabled", False)

        if self.use_contrastive:
            self.contrastive_loss_weight = contrastive_cfg.get("loss_weight", 0.1)
            self.annotation_loader = AnnotationLoader(contrastive_cfg["annotation_path"])
            self.clip_loss = ProteinFunctionCLIPLoss(
                structure_dim=model_cfg.encoder.d_out,
                text_model_name=contrastive_cfg.get("text_model", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"),
                shared_dim=contrastive_cfg.get("shared_dim", 512),
                learnable_temperature=contrastive_cfg.get("learnable_temperature", True),
                freeze_text_encoder=contrastive_cfg.get("freeze_text_encoder", True),
            )

        # Initialize forward folding if configured
        forward_folding_cfg = model_cfg.get("forward_folding", None)
        self.use_forward_folding = forward_folding_cfg is not None and forward_folding_cfg.get("enabled", False)

        if self.use_forward_folding:
            self.forward_folding_loss_weight = forward_folding_cfg.get("loss_weight", 1.0)
            self.forward_folding_skip_binned = forward_folding_cfg.get("skip_binned_losses", False)
            seq_enc_cfg = forward_folding_cfg.sequence_encoder

            assert seq_enc_cfg.d_out == model_cfg.encoder.d_out, \
                f"Sequence encoder d_out ({seq_enc_cfg.d_out}) must match structure encoder d_out ({model_cfg.encoder.d_out})"

            self.sequence_encoder = VanillaSequenceTokenEncoder(
                vocab_size=seq_enc_cfg.vocab_size,
                d_model=seq_enc_cfg.d_model,
                n_heads=seq_enc_cfg.n_heads,
                n_layers=seq_enc_cfg.n_layers,
                d_out=seq_enc_cfg.d_out,
            )

        # Initialize sequence decoder if configured (for sequence reconstruction in stages 2+)
        seq_decoder_cfg = model_cfg.get("sequence_decoder", None)
        self.use_sequence_decoder = seq_decoder_cfg is not None and seq_decoder_cfg.get("enabled", False)

        if self.use_sequence_decoder:
            self.sequence_decoder_loss_weight = seq_decoder_cfg.get("loss_weight", 1.0)
            self.sequence_decoder = VanillaSequenceTokenDecoder(
                encoder_d_out=model_cfg.encoder.d_out,
                d_model=seq_decoder_cfg.get("d_model", 512),
                n_heads=seq_decoder_cfg.get("n_heads", 8),
                n_layers=seq_decoder_cfg.get("n_layers", 4),
                vocab_size=seq_decoder_cfg.get("vocab_size", 33),
            )

        # Initialize cross-modal alignment if configured (for stages 2+)
        alignment_cfg = model_cfg.get("alignment", None)
        self.use_alignment = alignment_cfg is not None and alignment_cfg.get("enabled", False)

        if self.use_alignment:
            self.alignment_loss_weight = alignment_cfg.get("loss_weight", 0.5)
            self.alignment_loss = CrossModalAlignmentLoss(
                temperature=alignment_cfg.get("temperature", 0.1),
                symmetric=alignment_cfg.get("symmetric", True),
            )

        # Apply stage-specific freezing
        if self.training_stage is not None:
            self._apply_stage_freezing()

    def forward(self, input_list, use_as_tokenizer=False):
        self._step_count += 1

        coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list
        sequence_id = None
        """
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        residue_index: [B, L]
        seq_residue_tokens: [B, L]
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask # NOTE: due to data loading processing
        attention_mask = attention_mask.bool()

        # coords: torch.Tensor,
        # attention_mask: torch.Tensor | None = None,
        # sequence_id: torch.Tensor | None = None,
        # residue_index: torch.Tensor | None = None,
        z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index)
        assert self.quantizer.codebook_embed_size == self.encoder.d_out
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z, modality="structure")
        assert not z.isnan().any() and not quantized_indices.isnan().any()
        if use_as_tokenizer:
            return quantized_z, quantized_indices, z

        # Handle multi-codebook indices: MCQ returns [B, num_codebooks, L], standard returns [B, L]
        # The decoder needs [B, L] for structure_tokens argument
        if quantized_indices.dim() == 3:
            # MCQ: use first codebook indices for decoder (shape purposes)
            decoder_structure_tokens = quantized_indices[:, 0, :]
        else:
            decoder_structure_tokens = quantized_indices

        decoded_states = self.decoder.decode(quantized_z, decoder_structure_tokens, attention_mask, sequence_id)

        # reconstructed proteins
        bb_pred = decoded_states["bb_pred"]
        bb_rmsd_list, lddt_list = [], []
        for i in range(len(bb_pred)):
            pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[i].detach())
            pdb_chain_recon = pdb_chain_recon[:len(pdb_chain[i])]
        
            bb_rmsd = pdb_chain_recon.rmsd(pdb_chain[i], only_compute_backbone_rmsd=True)
            lddt = np.array(pdb_chain_recon.lddt_ca(pdb_chain[i]))
            bb_rmsd_list.append(bb_rmsd)
            lddt_list.append(lddt.mean())

        # reconstruction loss: 
        coords_recon = decoded_states["bb_pred"]
        # (1) backbone geometric distance loss: pairwise L2 distance matrix for 
        # the predicted and true coordinates of the 3 backbone atoms (N, Cα, C)
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            coords_recon, coords[:, :, :3, :], attention_mask) # [B, L, 3, 3]
        # (2) backbone geometric direction loss
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(
            coords_recon, coords[:, :, :3, :], attention_mask)
        # (3) backbone binned distance classification
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(
            decoded_states["pairwise_dist_logits"], coords, attention_mask)
        # (4) backbone binned direction classification
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            decoded_states["pairwise_dir_logits"], coords[:, :, :3, :], attention_mask)
        # (5) inverse folding 
        inverse_folding_loss, inverse_folding_metrics = self.compute_inverse_folding(
            decoded_states["last_hidden_state"], seq_residue_tokens, attention_mask)

        reconstruction_loss = (geom_dist_loss + geom_dir_loss + binned_dist_loss
                                + binned_dir_loss + inverse_folding_loss).mean()
        loss = reconstruction_loss * self.loss_weight["reconstruction_loss_weight"] + partial_loss

        # Contrastive loss for aligning structure with functional annotations
        contrastive_metrics = {}
        if self.use_contrastive:
            # Extract chain IDs from pdb_chain objects (format: "pdbid_chainid")
            chain_ids = [f"{pc.id}_{pc.chain_id}" for pc in pdb_chain]

            # Get function texts and annotation mask
            texts, ann_mask = self.annotation_loader.get_batch_annotations(chain_ids)
            ann_mask = ann_mask.to(coords.device)

            # Compute contrastive loss on encoder output z
            contrastive_loss, contrastive_metrics = self.clip_loss(
                z, attention_mask, texts, ann_mask
            )
            loss = loss + self.contrastive_loss_weight * contrastive_loss

        # Forward folding path (if enabled)
        forward_folding_metrics = {}
        forward_folding_loss = torch.tensor(0.0, device=coords.device)

        if self.use_forward_folding:
            # Encode sequence
            z_seq = self.sequence_encoder.encode(seq_residue_tokens, attention_mask, sequence_id)

            # Check if modality-specific codebook routing is configured
            has_modality_routing = (
                hasattr(self.quantizer, 'sequence_codebooks') and
                self.quantizer.sequence_codebooks is not None
            )

            if has_modality_routing:
                # Modality-specific routing: sequence-dedicated + shared codebooks receive gradients
                # Structure-dedicated codebooks are used in inference mode (no gradient updates)
                quantized_z_seq, quantized_indices_seq, partial_loss_seq, partial_metrics_seq = self.quantizer(
                    z_seq, modality="sequence"
                )
            else:
                # Option C: Stop ALL gradients to codebooks for sequence path
                # The sequence encoder learns to produce embeddings that match the
                # structure-optimized codebooks, but doesn't modify the codebooks themselves
                with torch.no_grad():
                    # Get indices without updating codebooks
                    quantized_indices_seq = self.quantizer.embedding2indices(z_seq)
                    # Look up embeddings from codebooks (detached)
                    quantized_z_seq_detached = self.quantizer.indices2embedding(quantized_indices_seq)

                # Straight-through estimator: gradient flows to z_seq (sequence encoder)
                # but not to codebooks. Forward pass uses quantized values.
                quantized_z_seq = z_seq + (quantized_z_seq_detached - z_seq).detach()

                # Compute commitment loss for sequence encoder
                commitment_loss_seq = F.mse_loss(z_seq, quantized_z_seq_detached.detach())

                # Metrics for sequence quantization path
                partial_loss_seq = commitment_loss_seq * self.loss_weight.get("commitment_loss_weight", 0.25)
                partial_metrics_seq = {
                    "commitment_loss": commitment_loss_seq,
                    "vocab_usage": torch.tensor(0.0, device=coords.device),
                }

            # Get decoder structure tokens
            if quantized_indices_seq.dim() == 3:
                decoder_structure_tokens_seq = quantized_indices_seq[:, 0, :]
            else:
                decoder_structure_tokens_seq = quantized_indices_seq

            # Decode sequence embeddings to predict structure
            # MEMORY OPTIMIZATION: Run decoder without storing activations for backward pass.
            # The decoder is shared and already trained via the structure reconstruction path.
            # The sequence encoder is trained via commitment loss (encouraging z_seq to match
            # codebook entries). The structure prediction here serves as a monitoring metric
            # but doesn't directly backprop through the decoder to save memory.
            with torch.no_grad():
                decoded_states_seq = self.decoder.decode(
                    quantized_z_seq.detach(), decoder_structure_tokens_seq, attention_mask, sequence_id,
                    skip_pairwise=self.forward_folding_skip_binned
                )

                # Compute forward folding losses (for monitoring, not gradient computation)
                coords_pred_from_seq = decoded_states_seq["bb_pred"]

                ff_geom_dist_loss, ff_geom_dist_metrics = self.compute_geometric_distance(
                    coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone()
                )
                ff_geom_dir_loss, ff_geom_dir_metrics = self.compute_geometric_direction(
                    coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone()
                )

                # Optionally skip binned losses to save memory
                if self.forward_folding_skip_binned:
                    ff_binned_dist_loss = torch.tensor(0.0, device=coords.device)
                    ff_binned_dir_loss = torch.tensor(0.0, device=coords.device)
                    ff_binned_dist_metrics = {}
                    ff_binned_dir_metrics = {}
                else:
                    ff_binned_dist_loss, ff_binned_dist_metrics = self.compute_binned_distance(
                        decoded_states_seq["pairwise_dist_logits"], coords.clone(), attention_mask.clone()
                    )
                    ff_binned_dir_loss, ff_binned_dir_metrics = self.compute_binned_direction(
                        decoded_states_seq["pairwise_dir_logits"], coords[:, :, :3, :].clone(), attention_mask.clone()
                    )

            # Structure prediction loss (monitoring only, computed in no_grad block above)
            ff_structure_loss = (
                ff_geom_dist_loss + ff_geom_dir_loss + ff_binned_dist_loss + ff_binned_dir_loss
            ).mean()

            # The actual trainable loss is the commitment loss from the sequence path.
            # This trains the sequence encoder to produce embeddings close to codebook entries.
            # The structure prediction metrics above monitor whether those entries lead to
            # good structure predictions (the decoder is trained via the structure path).
            forward_folding_loss = partial_loss_seq

            # Prefix all metrics with ff_
            forward_folding_metrics = {
                f"ff_{k}": v for k, v in {
                    **ff_geom_dist_metrics,
                    **ff_geom_dir_metrics,
                    **ff_binned_dist_metrics,
                    **ff_binned_dir_metrics,
                }.items()
            }
            forward_folding_metrics.update({f"ff_seq_{k}": v for k, v in partial_metrics_seq.items()})
            forward_folding_metrics["ff_structure_loss"] = ff_structure_loss  # For monitoring
            forward_folding_metrics["forward_folding_loss"] = forward_folding_loss

            # Add forward folding to total loss (only commitment loss contributes gradients)
            loss = loss + self.forward_folding_loss_weight * forward_folding_loss

        # Sequence reconstruction (if enabled, for stages 2+)
        seq_recon_metrics = {}
        if self.use_sequence_decoder and self.use_forward_folding:
            # Use the sequence encoder's quantized output for sequence reconstruction
            seq_recon_loss, seq_recon_metrics = self.compute_sequence_reconstruction(
                quantized_z_seq, seq_residue_tokens, attention_mask, sequence_id
            )
            loss = loss + self.sequence_decoder_loss_weight * seq_recon_loss

        # Cross-modal alignment (if enabled, for stages 2+)
        alignment_metrics = {}
        if self.use_alignment and self.use_forward_folding and hasattr(self.quantizer, 'get_soft_codebook_logits'):
            # Get soft logits from both structure and sequence encoders
            alignment_temp = self.model_cfg.get("alignment", {}).get("temperature", 0.1)

            # Structure encoder soft logits (detached - we align sequence TO structure)
            with torch.no_grad():
                structure_soft_logits = self.quantizer.get_soft_codebook_logits(z, temperature=alignment_temp)

            # Sequence encoder soft logits (receives gradients)
            sequence_soft_logits = self.quantizer.get_soft_codebook_logits(z_seq, temperature=alignment_temp)

            # Compute alignment loss
            alignment_loss, alignment_metrics = self.alignment_loss(
                source_logits=sequence_soft_logits,
                target_logits=structure_soft_logits,
                attention_mask=attention_mask,
            )
            loss = loss + self.alignment_loss_weight * alignment_loss

        metrics = {
            **geom_dist_metrics,
            **geom_dir_metrics,
            **binned_dist_metrics,
            **binned_dir_metrics,
            **inverse_folding_metrics,
            **partial_metrics,
            **contrastive_metrics,
            **forward_folding_metrics,
            **seq_recon_metrics,
            **alignment_metrics,
            "reconstruction_loss": reconstruction_loss,
            "bb_rmsd": torch.tensor(bb_rmsd_list, device=coords.device).mean(),
            "lddt": torch.tensor(lddt_list, device=coords.device).mean(),
        }
        loss_and_metrics = (loss, metrics)

        return (loss_and_metrics, )
    
    def compute_geometric_distance(self, x_recon, x, attention_mask, clamp_value=25):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        """
        assert x_recon.shape[-2] == 3 and x_recon.shape[-1] == 3
        
        # ignore padding regions
        x_recon[~attention_mask] = 0
        x[~attention_mask] = 0
        B, L, E = x.shape[0], x.shape[1], x.shape[-1]
        x_recon, x = x_recon.reshape(B, -1, E), x.reshape(B, -1, E) # [B, L, 3, 3] -> [B, L * 3, 3] 

        dist_pred = torch.cdist(x_recon, x_recon, p=2.0) # [B, L * 3, L * 3]
        dist_true = torch.cdist(x, x, p=2.0)

        dist_mask = attention_mask.repeat(1, 3)
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, L * 3, L * 3]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dist_loss": loss.mean(),
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return loss.mean(), metric

    def compute_direction_vectors(self, coords,):
        """
        coords: [B, L, 3, 3]
        """
        # N -> Ca
        v1 = coords[:, :, 1, :] - coords[:, :, 0, :] # [B, 0~L, 3]
        # Ca -> C
        v2 = coords[:, :, 2, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # C -> N_next
        v3 = coords[:, 1:, 0, :] - coords[:, :-1, 2, :] # [B, 0~L-1, 3]
        # -(N -> Ca) x (Ca -> C)
        v4 = - torch.cross(v1, v2, dim=-1) # [B, 0~L, 3]
        # (C_prev -> N) x (N -> Ca)
        tmp = coords[:, 1:, 0, :] - coords[:, :-1, 2, :] # [B, 1~L, 3]
        v5 = torch.cross(tmp, v1[:, 1:], dim=-1)
        # (Ca -> C) x (C -> N_next)
        v6 = torch.cross(v2[:, :-1], v3, dim=-1) # [B, 0~L-1, 3]
        
        ret = [v1[:, 1:-1], v2[:, 1:-1], v3[:, 1:], v4[:, 1:-1], v5[:, :-1], v6[:, 1:]] # [B, L-2, 3]
        ret = torch.stack(ret, dim=1) # [B, 6, L-2, 3]
        ret = ret.reshape(ret.shape[0], -1, ret.shape[-1]) # [B, 6 * (L-2), 3]

        return ret


    def compute_geometric_direction(self, x_recon, x, attention_mask, clamp_value=20):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        vec_pred = self.compute_direction_vectors(x_recon)
        vec = self.compute_direction_vectors(x)

        # pairwise dot product
        dist_pred = torch.matmul(vec_pred, torch.transpose(vec_pred, 1, 2)) # [B, 6(L-2), 6(L-2)]
        dist_true = torch.matmul(vec, torch.transpose(vec, 1, 2)) # [B, 6(L-2), 6(L-2)]

        dist_mask = attention_mask[:, 1:-1].repeat(1, 6) # [B, 6(L-2)]
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, 6(L-2), 6(L-2)]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dir_loss": loss.mean(),
            f"geom_dir_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dir_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        return loss.mean(), metric

    def compute_binned_direction(self, pairwise_logits, coords, attention_mask):
        """
        pairwise_logits: [B, L, L, 96]
        coords: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        # compute from ground truth
        # unit vectors
        # Ca -> C
        v1 = coords[:, :, 2, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # Ca -> N
        v2 = coords[:, :, 0, :] - coords[:, :, 1, :] # [B, 0~L, 3]
        # (Ca -> C) x (Ca -> N)
        v3 = torch.cross(v1, v2, dim=-1) # [B, L, 3]
        v1 = F.normalize(v1, p=2, dim=-1)
        v2 = F.normalize(v2, p=2, dim=-1)
        v3 = F.normalize(v3, p=2, dim=-1)

        # dot products
        pairwise_prod = torch.stack([
            torch.matmul(v1, torch.transpose(v2, 1, 2)), # [B, L, L]
            torch.matmul(v1, torch.transpose(v3, 1, 2)),
            torch.matmul(v2, torch.transpose(v1, 1, 2)),
            torch.matmul(v2, torch.transpose(v3, 1, 2)),
            torch.matmul(v3, torch.transpose(v1, 1, 2)),
            torch.matmul(v3, torch.transpose(v2, 1, 2)),
        ], dim=-1) # [B, L, L, 6]
        NUM_BIN = 16
        bin_edges = [-1 + 0.125 * i for i in range(NUM_BIN)] + [1]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(pairwise_prod, bin_edges, right=True) - 1 # [B, L, L, 6]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        pairwise_logits = pairwise_logits.reshape([_ for _ in binned_labels.shape] + [-1]) # [B, L, L, 6, NUM_BIN]

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask].reshape(-1, NUM_BIN), binned_labels[mask].reshape(-1)
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)
        
        metric = {
            f"binned_dir_loss": loss.mean(),
            f"binned_dir_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_binned_distance(self, pairwise_logits, coords, attention_mask):
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        """

        # calculate Cbeta
        cbeta = infer_cbeta_from_atom37(coords) # [B, L, 3]

        # pairwise Cbeta distance
        NUM_BIN = 64
        dist_true = torch.cdist(cbeta, cbeta, p=2.0)
        bin_edges = [0] + [(2.3125 + 0.3075 * i) ** 2 for i in range(NUM_BIN)]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(dist_true, bin_edges, right=True) - 1 # [B, L, L]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        assert binned_labels.min() >= 0 and binned_labels.max() < NUM_BIN

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)
        
        metric = {
            f"binned_dist_loss": loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_inverse_folding(self, h, residue_labels, attention_mask):
        """
        h: [B, L, d_model=1024]
        residue_labels: [B, L]
        attention_mask: [B, L]
        """
        logits = self.inverse_folding_head(h) # [B, L, num_AAs]

        if not (logits.shape[0] == attention_mask.shape[0] and logits.shape[1] == attention_mask.shape[1]):
            raise ValueError

        logits, residue_labels = logits[attention_mask], residue_labels[attention_mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, residue_labels)

        metric = {
            f"inverse_folding_loss": loss.mean(),
            f"inverse_folding_accuracy": (logits.argmax(dim=-1) == residue_labels).float().mean(),
        }
        return loss.mean(), metric

    def compute_sequence_reconstruction(self, quantized_z, seq_residue_tokens, attention_mask, sequence_id=None):
        """
        Compute sequence reconstruction loss from quantized embeddings.

        Args:
            quantized_z: [B, L, d_out] quantized embeddings
            seq_residue_tokens: [B, L] target amino acid tokens
            attention_mask: [B, L] boolean mask
            sequence_id: [B, L] optional sequence IDs

        Returns:
            loss: scalar reconstruction loss
            metrics: dictionary with accuracy and other metrics
        """
        if not self.use_sequence_decoder:
            return torch.tensor(0.0, device=quantized_z.device), {}

        decoded_seq = self.sequence_decoder.decode(quantized_z, attention_mask, sequence_id)
        logits = decoded_seq["logits"]  # [B, L, vocab_size]

        # Flatten for loss computation
        logits_flat = logits[attention_mask]  # [N, vocab_size]
        targets_flat = seq_residue_tokens[attention_mask]  # [N]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits_flat, targets_flat)

        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == targets_flat).float().mean()

        metrics = {
            "seq_recon_loss": loss.mean(),
            "seq_recon_accuracy": accuracy,
        }

        return loss.mean(), metrics

    def _apply_stage_freezing(self):
        """
        Apply component freezing based on training stage.

        Stage 1: Train structure encoder/decoder, codebook, inverse folding
        Stage 2: Freeze structure path, train sequence encoder/decoder, alignment
        Stage 3: Unfreeze all (joint fine-tuning with lower LR for structure path)
        Stage 4: Initially freeze stages 1-3, train function encoder/decoder
        """
        stage = self.training_stage

        if stage == 1:
            # Stage 1: Structure foundation - everything trains normally
            pass

        elif stage == 2:
            # Stage 2: Freeze structure encoder, decoder, and codebook
            # Train sequence encoder, sequence decoder, alignment
            self._freeze_module(self.encoder)
            self._freeze_module(self.decoder)
            self._freeze_module(self.quantizer)
            self._freeze_module(self.inverse_folding_head)

        elif stage == 3:
            # Stage 3: Joint fine-tuning - everything unfrozen
            # LR adjustments handled in get_parameter_groups()
            pass

        elif stage == 4:
            # Stage 4: Function integration
            # Initially freeze structure/sequence paths, train function
            # (can be unfrozen later in training)
            self._freeze_module(self.encoder)
            self._freeze_module(self.decoder)
            self._freeze_module(self.quantizer)
            if self.use_forward_folding:
                self._freeze_module(self.sequence_encoder)
            if self.use_sequence_decoder:
                self._freeze_module(self.sequence_decoder)

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module: nn.Module):
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, base_lr: float = 1e-4):
        """
        Get parameter groups with stage-specific learning rates.

        For Stage 3 (joint fine-tuning), structure path gets lower LR.

        Args:
            base_lr: Base learning rate

        Returns:
            List of parameter group dicts for optimizer
        """
        stage = self.training_stage

        if stage == 3:
            # Stage 3: Lower LR for structure path (0.1x), normal LR for sequence path
            structure_params = []
            sequence_params = []
            other_params = []

            structure_modules = [self.encoder, self.decoder, self.quantizer, self.inverse_folding_head]
            sequence_modules = []
            if self.use_forward_folding:
                sequence_modules.append(self.sequence_encoder)
            if self.use_sequence_decoder:
                sequence_modules.append(self.sequence_decoder)
            if self.use_alignment:
                sequence_modules.append(self.alignment_loss)

            structure_param_set = set()
            for module in structure_modules:
                for param in module.parameters():
                    structure_param_set.add(id(param))
                    if param.requires_grad:
                        structure_params.append(param)

            sequence_param_set = set()
            for module in sequence_modules:
                for param in module.parameters():
                    sequence_param_set.add(id(param))
                    if param.requires_grad:
                        sequence_params.append(param)

            # Other params (contrastive, etc.)
            for param in self.parameters():
                param_id = id(param)
                if param_id not in structure_param_set and param_id not in sequence_param_set:
                    if param.requires_grad:
                        other_params.append(param)

            return [
                {"params": structure_params, "lr": base_lr * 0.1, "name": "structure_path"},
                {"params": sequence_params, "lr": base_lr, "name": "sequence_path"},
                {"params": other_params, "lr": base_lr, "name": "other"},
            ]

        else:
            # Default: all params get same LR
            return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr}]

    def get_frozen_components(self) -> list[str]:
        """Return list of component names that are currently frozen."""
        frozen = []
        components = [
            ("encoder", self.encoder),
            ("decoder", self.decoder),
            ("quantizer", self.quantizer),
            ("inverse_folding_head", self.inverse_folding_head),
        ]

        if self.use_forward_folding:
            components.append(("sequence_encoder", self.sequence_encoder))
        if self.use_sequence_decoder:
            components.append(("sequence_decoder", self.sequence_decoder))
        if self.use_contrastive:
            components.append(("clip_loss", self.clip_loss))
        if self.use_alignment:
            components.append(("alignment_loss", self.alignment_loss))

        for name, module in components:
            all_frozen = all(not p.requires_grad for p in module.parameters())
            if all_frozen and len(list(module.parameters())) > 0:
                frozen.append(name)

        return frozen


class LightningVQPretrainModel(pl.LightningModule):
    """
    PTL wrapper class for VQ-VAE pre-training
    """

    def __init__(
        self,
        model_cfg,
        trainer,
        py_logger,
        optimizer_cfg,
        all_split_names,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.trainer = trainer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
    
        self.all_split_names = all_split_names
        for split in self.all_split_names:
            setattr(self, f"{split}_step_outputs", [])

    def setup(self, stage: str):
        """
        Set up the module, including model creation
        Args:
            stage: PTL stage train/val/test can be used to induce different
                    behavior only used for inheritance
        """

        self.trainer.strategy.config["train_micro_batch_size_per_gpu"] = self.optimizer_cfg.micro_batch_size
        self.model = model_init_fn(self.trainer, self.model_cfg)

        # get time here for first iteration at batch 0
        # logged in on_train_batch_end
        self._last_logged_batch_start_time = time.monotonic()

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_list"])
        loss, metrics = outputs[0]

        self.log(
            "training_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size, logger=True, sync_dist=True,
        )

        # Log key metrics for monitoring
        metrics_to_log = [
            "reconstruction_loss", "bb_rmsd", "lddt", "vocab_usage",
            "ff_structure_loss", "forward_folding_loss", "ff_seq_commitment_loss",
            "contrastive_loss", "contrastive_accuracy",
            # Multi-stage training metrics
            "seq_recon_loss", "seq_recon_accuracy",
            "align_kl_loss", "align_token_agreement_rate",
        ]
        for key in metrics_to_log:
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                if hasattr(value, 'item'):
                    value = value.item()
                self.log(
                    f"train/{key}", value, on_step=True, on_epoch=False,
                    batch_size=self.optimizer_cfg.micro_batch_size, logger=True, sync_dist=True,
                )

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log time/step and TFLOPS
        Args:
            outputs: outputs of train_step, not used, required for hook
            batch: use batch to get input/output sequence length for TFLOPs
            batch_idx: batch number, not used required for hook
        """

        if batch_idx > 0 and batch_idx % self.trainer.log_every_n_steps == 0:
            # get the time for this iteration
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            # start timeer for the next iteration
            self._last_logged_batch_start_time = time.monotonic()
            time_per_step = elapsed_time / self.trainer.log_every_n_steps

            # useful to log this even though PTL provides it in the progressbar
            # PTL logs provide exponential decaying average which is not useful
            # forquick benchmarking, especially for large models
            self.log(
                "sec/step", time_per_step, on_step=True, prog_bar=True, 
                logger=True, rank_zero_only=True,
            )
        
        torch.cuda.empty_cache()

    def _valid_or_test_step(self, batch, batch_idx, split="validation"):
        outputs = self.model(batch["input_list"])
        loss, metrics = outputs[0]

        log_metrics = {
            f"{split}_{k}": v for k, v in metrics.items()
        }

        self.log_dict(
            {f"{split}_loss": loss, **log_metrics},
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )

        return {
            f"{split}_loss": loss,
            **log_metrics,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = self.all_split_names[dataloader_idx]
        outputs = self._valid_or_test_step(batch, batch_idx, split=split)
        getattr(self, f"{split}_step_outputs").append(outputs)
        return outputs

    def on_train_start(self):
        # override the lambda schedulers
        # default configs do not adjust the schedulers
        self.lr_schedulers().lr_lambdas = [
            lambda x: self.optimizer_cfg.override.mult_factor
            * fn(x + self.optimizer_cfg.override.add_index)
            for fn in self.lr_schedulers().lr_lambdas
        ]

    def _valid_or_test_epoch_end(self, outputs, split="validation"):
        
        agg_result = {k: [] for k in outputs[0].keys() if k.startswith(split)}
        for out in outputs:
            for k in out.keys():
                if k.startswith(split):
                    agg_result[k].append(out[k])

        for k in agg_result.keys():
            agg_result[k] = torch.stack(agg_result[k]).mean()

        self.log_dict(
            agg_result, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size, add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self):
        for split in self.all_split_names:
            self._valid_or_test_epoch_end(getattr(self, f"{split}_step_outputs"), split=split)
        for split in self.all_split_names:
            getattr(self, f"{split}_step_outputs").clear()

    def on_before_optimizer_step(self, optimizer):
        for n,p in self.model.named_parameters():
            grad_data = deepspeed.utils.safe_get_full_grad(p)
            p.grad = grad_data
        norms = grad_norm(self.model, norm_type=2)
        norms = {k:v.to(grad_data.device) for k,v in norms.items()}
        
        self.log_dict(
            norms, prog_bar=True, sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size, add_dataloader_idx=False,
            #on_step=True, #on_epoch=True,
        )

    def configure_optimizers(self):
        # hyperparameter logging needs to occur after ddp launch
        # inside config_optimizers since this occurs after ddp launch
        # use trainer logger which ensures it is mstar logger
        # self.trainer.logger.log_hyperparams(self.full_experiment_config)

        # Check if using multi-stage training with stage-specific LRs
        training_stage = self.model_cfg.get("training_stage", None)
        base_lr = self.optimizer_cfg.optimizer.lr

        if training_stage == 3 and hasattr(self.model, 'get_parameter_groups'):
            # Stage 3: Use model's parameter groups for differential LRs
            stage_param_groups = self.model.get_parameter_groups(base_lr)

            # Apply weight decay settings to stage param groups
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            decay_parameters = [name for name in decay_parameters if "layer_norm" not in name]
            decay_parameters = [name for name in decay_parameters if "layernorm" not in name]

            param_groups = []
            for group in stage_param_groups:
                group_params = group["params"]
                group_lr = group.get("lr", base_lr)
                group_name = group.get("name", "default")

                # Split into decay/no-decay within this group
                decay_params = []
                nodecay_params = []
                for p in group_params:
                    # Find param name
                    param_name = None
                    for n, param in self.model.named_parameters():
                        if param is p:
                            param_name = n
                            break
                    if param_name and any(nd in param_name for nd in decay_parameters):
                        decay_params.append(p)
                    else:
                        nodecay_params.append(p)

                if decay_params:
                    param_groups.append({
                        "params": decay_params,
                        "lr": group_lr,
                        "weight_decay": self.optimizer_cfg.optimizer.weight_decay,
                    })
                if nodecay_params:
                    param_groups.append({
                        "params": nodecay_params,
                        "lr": group_lr,
                        "weight_decay": 0.0,
                    })
        else:
            # Default: standard parameter groups
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            # filter out bias
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # filter out layernorm with a variety of spellings
            decay_parameters = [name for name in decay_parameters if "layer_norm" not in name]
            decay_parameters = [name for name in decay_parameters if "layernorm" not in name]

            params_decay = [p for n, p in self.model.named_parameters() if (any(nd in n for nd in decay_parameters)) and p.requires_grad]
            params_nodecay = [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in decay_parameters)) and p.requires_grad]

            param_groups = [
                {
                    "params": params_decay,
                    "weight_decay": self.optimizer_cfg.optimizer.weight_decay,
                },
                {
                    "params": params_nodecay,
                    "weight_decay": 0.0
                },
            ]

        optimizer = get_optimizer(param_groups, self.optimizer_cfg.optimizer)

        scheduler = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=optimizer)
        return (
            [optimizer],
            [{
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "validation_loss",
            }],
        )
