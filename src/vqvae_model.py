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

        # pLDDT-weighted loss configuration
        self.use_plddt_weighting = model_cfg.get("use_plddt_weighting", False)

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

    def forward(self, input_list, use_as_tokenizer=False):
        self._step_count += 1

        # Support both old format (5 elements) and new format with pLDDT (6 elements)
        if len(input_list) == 6:
            coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain, plddt = input_list
        else:
            coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list
            plddt = None  # No pLDDT weighting
        sequence_id = None
        """
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        residue_index: [B, L]
        seq_residue_tokens: [B, L]
        plddt: [B, L] (optional) - confidence scores for loss weighting
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
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
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

        # Determine pLDDT weights for loss computation (only if enabled in config)
        plddt_weights = plddt if (self.use_plddt_weighting and plddt is not None) else None

        # (1) backbone geometric distance loss: pairwise L2 distance matrix for
        # the predicted and true coordinates of the 3 backbone atoms (N, Cα, C)
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            coords_recon, coords[:, :, :3, :], attention_mask, plddt_weights=plddt_weights) # [B, L, 3, 3]
        # (2) backbone geometric direction loss
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(
            coords_recon, coords[:, :, :3, :], attention_mask, plddt_weights=plddt_weights)
        # (3) backbone binned distance classification
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(
            decoded_states["pairwise_dist_logits"], coords, attention_mask, plddt_weights=plddt_weights)
        # (4) backbone binned direction classification
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            decoded_states["pairwise_dir_logits"], coords[:, :, :3, :], attention_mask, plddt_weights=plddt_weights)
        # (5) inverse folding
        inverse_folding_loss, inverse_folding_metrics = self.compute_inverse_folding(
            decoded_states["last_hidden_state"], seq_residue_tokens, attention_mask, plddt_weights=plddt_weights)

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

            # Option C: Stop gradients to codebooks for sequence path
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

            # Compute commitment loss for sequence encoder (encourages z_seq to be close to codebook entries)
            # Note: Only z_seq receives gradients here, not the codebooks
            commitment_loss_seq = F.mse_loss(z_seq, quantized_z_seq_detached.detach())

            # Metrics for sequence quantization path
            partial_loss_seq = commitment_loss_seq * self.loss_weight.get("commitment_loss_weight", 0.25)
            partial_metrics_seq = {
                "commitment_loss": commitment_loss_seq,
                "vocab_usage": torch.tensor(0.0, device=coords.device),  # Not tracking for seq path
            }

            # Get decoder structure tokens
            if quantized_indices_seq.dim() == 3:
                decoder_structure_tokens_seq = quantized_indices_seq[:, 0, :]
            else:
                decoder_structure_tokens_seq = quantized_indices_seq

            # Decode sequence embeddings to predict structure
            # Skip pairwise head if binned losses are skipped to save memory
            decoded_states_seq = self.decoder.decode(
                quantized_z_seq, decoder_structure_tokens_seq, attention_mask, sequence_id,
                skip_pairwise=self.forward_folding_skip_binned
            )

            # Compute forward folding losses
            coords_pred_from_seq = decoded_states_seq["bb_pred"]

            ff_geom_dist_loss, ff_geom_dist_metrics = self.compute_geometric_distance(
                coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone(),
                plddt_weights=plddt_weights
            )
            ff_geom_dir_loss, ff_geom_dir_metrics = self.compute_geometric_direction(
                coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone(),
                plddt_weights=plddt_weights
            )

            # Optionally skip binned losses to save memory
            if self.forward_folding_skip_binned:
                ff_binned_dist_loss = torch.tensor(0.0, device=coords.device)
                ff_binned_dir_loss = torch.tensor(0.0, device=coords.device)
                ff_binned_dist_metrics = {}
                ff_binned_dir_metrics = {}
            else:
                ff_binned_dist_loss, ff_binned_dist_metrics = self.compute_binned_distance(
                    decoded_states_seq["pairwise_dist_logits"], coords.clone(), attention_mask.clone(),
                    plddt_weights=plddt_weights
                )
                ff_binned_dir_loss, ff_binned_dir_metrics = self.compute_binned_direction(
                    decoded_states_seq["pairwise_dir_logits"], coords[:, :, :3, :].clone(), attention_mask.clone(),
                    plddt_weights=plddt_weights
                )

            # Combine forward folding losses
            forward_folding_loss = (
                ff_geom_dist_loss + ff_geom_dir_loss + ff_binned_dist_loss + ff_binned_dir_loss
            ).mean()

            # Add quantization losses from sequence path
            forward_folding_loss = forward_folding_loss + partial_loss_seq

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
            forward_folding_metrics["forward_folding_loss"] = forward_folding_loss

            # Add forward folding to total loss
            loss = loss + self.forward_folding_loss_weight * forward_folding_loss

        metrics = {
            **geom_dist_metrics,
            **geom_dir_metrics,
            **binned_dist_metrics,
            **binned_dir_metrics,
            **inverse_folding_metrics,
            **partial_metrics,
            **contrastive_metrics,
            **forward_folding_metrics,
            "reconstruction_loss": reconstruction_loss,
            "bb_rmsd": torch.tensor(bb_rmsd_list, device=coords.device).mean(),
            "lddt": torch.tensor(lddt_list, device=coords.device).mean(),
        }
        loss_and_metrics = (loss, metrics)
        
        return (loss_and_metrics, )
    
    def compute_geometric_distance(self, x_recon, x, attention_mask, plddt_weights=None, clamp_value=25):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        plddt_weights: [B, L] optional confidence weights (0-100 scale, will be normalized)
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

        # Compute pLDDT-based pairwise weights if provided
        if plddt_weights is not None:
            # Normalize pLDDT to [0, 1] and repeat for 3 backbone atoms
            weights = (plddt_weights / 100.0).repeat(1, 3)  # [B, L*3]
            # Pairwise weight = geometric mean of both residues' confidence
            pair_weights = torch.sqrt(weights.unsqueeze(-1) * weights.unsqueeze(1))  # [B, L*3, L*3]
            pair_weights = pair_weights[dist_mask]
        else:
            pair_weights = None

        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)

        # Apply pLDDT weighting
        if pair_weights is not None:
            weighted_loss = (loss * pair_weights).sum() / (pair_weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()

        metric = {
            f"geom_dist_loss": loss.mean(),  # Unweighted for comparison
            f"geom_dist_loss_weighted": weighted_loss if pair_weights is not None else loss.mean(),
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return weighted_loss, metric

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


    def compute_geometric_direction(self, x_recon, x, attention_mask, plddt_weights=None, clamp_value=20):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        attention_mask: [B, L]
        plddt_weights: [B, L] optional confidence weights (0-100 scale)
        """
        vec_pred = self.compute_direction_vectors(x_recon)
        vec = self.compute_direction_vectors(x)

        # pairwise dot product
        dist_pred = torch.matmul(vec_pred, torch.transpose(vec_pred, 1, 2)) # [B, 6(L-2), 6(L-2)]
        dist_true = torch.matmul(vec, torch.transpose(vec, 1, 2)) # [B, 6(L-2), 6(L-2)]

        dist_mask = attention_mask[:, 1:-1].repeat(1, 6) # [B, 6(L-2)]
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, 6(L-2), 6(L-2)]

        # Compute pLDDT-based pairwise weights if provided
        if plddt_weights is not None:
            # Trim pLDDT to match direction vectors (exclude first and last residue)
            plddt_trimmed = plddt_weights[:, 1:-1]  # [B, L-2]
            weights = (plddt_trimmed / 100.0).repeat(1, 6)  # [B, 6(L-2)]
            pair_weights = torch.sqrt(weights.unsqueeze(-1) * weights.unsqueeze(1))  # [B, 6(L-2), 6(L-2)]
            pair_weights = pair_weights[dist_mask]
        else:
            pair_weights = None

        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none") # flattened
        loss = torch.clamp(loss, max=clamp_value)

        # Apply pLDDT weighting
        if pair_weights is not None:
            weighted_loss = (loss * pair_weights).sum() / (pair_weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()

        metric = {
            f"geom_dir_loss": loss.mean(),  # Unweighted for comparison
            f"geom_dir_loss_weighted": weighted_loss if pair_weights is not None else loss.mean(),
            f"geom_dir_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dir_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        return weighted_loss, metric

    def compute_binned_direction(self, pairwise_logits, coords, attention_mask, plddt_weights=None):
        """
        pairwise_logits: [B, L, L, 96]
        coords: [B, L, 3, 3]
        attention_mask: [B, L]
        plddt_weights: [B, L] optional confidence weights (0-100 scale)
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

        # Compute pLDDT-based pairwise weights if provided
        if plddt_weights is not None:
            weights = plddt_weights / 100.0  # [B, L]
            pair_weights = torch.sqrt(weights.unsqueeze(-1) * weights.unsqueeze(1))  # [B, L, L]
            # Expand to match 6 direction pairs and flatten
            pair_weights = pair_weights.unsqueeze(-1).expand(-1, -1, -1, 6)  # [B, L, L, 6]
            pair_weights = pair_weights[mask].reshape(-1)
        else:
            pair_weights = None

        pairwise_logits, binned_labels = pairwise_logits[mask].reshape(-1, NUM_BIN), binned_labels[mask].reshape(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        # Apply pLDDT weighting
        if pair_weights is not None:
            weighted_loss = (loss * pair_weights).sum() / (pair_weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()

        metric = {
            f"binned_dir_loss": loss.mean(),  # Unweighted for comparison
            f"binned_dir_loss_weighted": weighted_loss if pair_weights is not None else loss.mean(),
            f"binned_dir_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return weighted_loss, metric

    def compute_binned_distance(self, pairwise_logits, coords, attention_mask, plddt_weights=None):
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        plddt_weights: [B, L] optional confidence weights (0-100 scale)
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

        # Compute pLDDT-based pairwise weights if provided
        if plddt_weights is not None:
            weights = plddt_weights / 100.0  # [B, L]
            pair_weights = torch.sqrt(weights.unsqueeze(-1) * weights.unsqueeze(1))  # [B, L, L]
            pair_weights = pair_weights[mask]
        else:
            pair_weights = None

        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        # Apply pLDDT weighting
        if pair_weights is not None:
            weighted_loss = (loss * pair_weights).sum() / (pair_weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()

        metric = {
            f"binned_dist_loss": loss.mean(),  # Unweighted for comparison
            f"binned_dist_loss_weighted": weighted_loss if pair_weights is not None else loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return weighted_loss, metric

    def compute_inverse_folding(self, h, residue_labels, attention_mask, plddt_weights=None):
        """
        h: [B, L, d_model=1024]
        residue_labels: [B, L]
        attention_mask: [B, L]
        plddt_weights: [B, L] optional confidence weights (0-100 scale)
        """
        logits = self.inverse_folding_head(h) # [B, L, num_AAs]

        if not (logits.shape[0] == attention_mask.shape[0] and logits.shape[1] == attention_mask.shape[1]):
            raise ValueError

        # Compute per-residue pLDDT weights if provided
        if plddt_weights is not None:
            weights = (plddt_weights / 100.0)[attention_mask]  # Normalize and flatten
        else:
            weights = None

        logits, residue_labels = logits[attention_mask], residue_labels[attention_mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, residue_labels)

        # Apply pLDDT weighting
        if weights is not None:
            weighted_loss = (loss * weights).sum() / (weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()

        metric = {
            f"inverse_folding_loss": loss.mean(),  # Unweighted for comparison
            f"inverse_folding_loss_weighted": weighted_loss if weights is not None else loss.mean(),
            f"inverse_folding_accuracy": (logits.argmax(dim=-1) == residue_labels).float().mean(),
        }
        return weighted_loss, metric
    

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
            # Convert floats to tensors for stacking
            values = agg_result[k]
            device = self.device
            tensor_values = [
                torch.tensor(v, device=device) if isinstance(v, (int, float)) else v.to(device)
                for v in values
            ]
            agg_result[k] = torch.stack(tensor_values).mean()

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

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        # filter out bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # filter out layernorm with a variety of spellings
        decay_parameters = [name for name in decay_parameters if "layer_norm" not in name]
        decay_parameters = [name for name in decay_parameters if "layernorm" not in name]
        
        params_decay = [p for n, p in self.model.named_parameters() if (any(nd in n for nd in decay_parameters))]
        params_nodecay = [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in decay_parameters))]
        
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
