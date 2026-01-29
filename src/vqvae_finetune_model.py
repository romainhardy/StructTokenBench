"""
Fine-tuning model for adapting pre-trained AminoAseed encoder to MCQ quantization.

This module provides AminoAseedMCQFinetune, which:
1. Loads a pre-trained encoder and freezes it
2. Adds an optional projection layer for L2 -> cosine similarity adaptation
3. Uses MCQ quantizer for improved codebook utilization
4. Supports warm-starting the decoder from pre-trained weights
"""

import torch
from torch import nn

from vqvae_model import VQVAEModel
from vqvae.quantizer_module import MCQQuantizer


class AminoAseedMCQFinetune(VQVAEModel):
    """
    Fine-tuning model: frozen encoder + projection + MCQ + trainable decoder.

    Inherits from VQVAEModel to reuse loss computation and forward pass logic.
    The key modifications are:
    - Pre-trained encoder weights are loaded and frozen
    - Optional projection layer adapts L2-trained encoder to cosine MCQ
    - MCQ quantizer replaces the standard quantizer
    """

    def __init__(self, model_cfg):
        # Initialize parent (creates encoder, decoder, quantizer)
        super().__init__(model_cfg)

        # Store config for later access
        self.finetune_cfg = model_cfg

        # Load pre-trained weights if path provided
        pretrained_ckpt_path = model_cfg.get("pretrained_ckpt_path", None)
        if pretrained_ckpt_path:
            self._load_pretrained(pretrained_ckpt_path)

        # Freeze encoder parameters
        if model_cfg.get("freeze_encoder", True):
            self._freeze_encoder()

        # Add projection layer if configured
        # The projection layer adapts encoder output (d_out) to MCQ dimension (codebook_embed_size)
        use_projection = model_cfg.get("use_projection", True)
        encoder_d_out = model_cfg.encoder.d_out
        mcq_dim = model_cfg.quantizer.codebook_embed_size

        if use_projection:
            if encoder_d_out != mcq_dim:
                # Project from encoder dimension to MCQ dimension
                self.projection = nn.Sequential(
                    nn.Linear(encoder_d_out, mcq_dim),
                    nn.LayerNorm(mcq_dim),
                )
                print(f"[AminoAseedMCQFinetune] Projection: {encoder_d_out} -> {mcq_dim}")
            else:
                # Same dimension, just add LayerNorm
                self.projection = nn.Sequential(
                    nn.Linear(encoder_d_out, mcq_dim),
                    nn.LayerNorm(mcq_dim),
                )
        else:
            if encoder_d_out != mcq_dim:
                raise ValueError(
                    f"use_projection=False but encoder d_out ({encoder_d_out}) != "
                    f"MCQ codebook_embed_size ({mcq_dim}). Enable projection layer."
                )
            self.projection = nn.Identity()

        # Add unprojection layer to go from MCQ dim back to decoder dim
        decoder_d_model = model_cfg.decoder.d_model
        if mcq_dim != decoder_d_model:
            self.unprojection = nn.Sequential(
                nn.Linear(mcq_dim, decoder_d_model),
                nn.LayerNorm(decoder_d_model),
            )
            print(f"[AminoAseedMCQFinetune] Unprojection: {mcq_dim} -> {decoder_d_model}")
        else:
            self.unprojection = nn.Identity()

    def _load_pretrained(self, ckpt_path: str):
        """
        Load encoder and optionally decoder from pre-trained checkpoint.

        The checkpoint format is expected to be DeepSpeed's model_states.pt,
        with keys prefixed by "model." (e.g., "model.encoder.transformer.blocks.0...")
        """
        state = torch.load(ckpt_path, map_location="cpu")

        # Handle different checkpoint formats
        if "module" in state:
            state = state["module"]

        # Extract encoder state dict
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("model.encoder."):
                new_key = k.replace("model.encoder.", "")
                encoder_state[new_key] = v
            elif k.startswith("encoder."):
                new_key = k.replace("encoder.", "")
                encoder_state[new_key] = v

        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                print(f"[AminoAseedMCQFinetune] Missing encoder keys: {missing}")
            if unexpected:
                print(f"[AminoAseedMCQFinetune] Unexpected encoder keys: {unexpected}")
            print(f"[AminoAseedMCQFinetune] Loaded {len(encoder_state)} encoder parameters")

        # Optionally warm-start decoder
        if self.finetune_cfg.get("warm_start_decoder", True):
            decoder_state = {}
            for k, v in state.items():
                if k.startswith("model.decoder."):
                    new_key = k.replace("model.decoder.", "")
                    decoder_state[new_key] = v
                elif k.startswith("decoder."):
                    new_key = k.replace("decoder.", "")
                    decoder_state[new_key] = v

            if decoder_state:
                missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
                if missing:
                    print(f"[AminoAseedMCQFinetune] Missing decoder keys: {missing}")
                if unexpected:
                    print(f"[AminoAseedMCQFinetune] Unexpected decoder keys: {unexpected}")
                print(f"[AminoAseedMCQFinetune] Loaded {len(decoder_state)} decoder parameters")

    def _freeze_encoder(self):
        """Freeze all encoder parameters to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"[AminoAseedMCQFinetune] Froze encoder ({sum(1 for _ in self.encoder.parameters())} parameters)")

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode coordinates to latent representation with projection.

        Overrides the parent's implicit encode to add the projection layer
        that adapts the L2-trained encoder output to MCQ's cosine similarity space.
        """
        # Get encoder output (this calls encoder.encode internally)
        z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index)

        # Apply projection to adapt to MCQ's cosine similarity space
        z = self.projection(z)

        return z

    def forward(self, input_list, use_as_tokenizer=False):
        """
        Forward pass with projection layer applied after encoding.

        This overrides the parent's forward to insert the projection layer
        between encoding and quantization.
        """
        self._step_count += 1

        # Support both 5-element (old format) and 6-element (with plddt) input
        if len(input_list) == 6:
            coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain, plddt = input_list
        else:
            coords, attention_mask, residue_index, seq_residue_tokens, pdb_chain = input_list
            plddt = None
        sequence_id = None

        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask  # NOTE: due to data loading processing
        attention_mask = attention_mask.bool()

        # Encode with projection layer
        z = self.encode(coords, attention_mask, sequence_id, residue_index)

        # Verify dimensions match (after projection, z should match quantizer dimension)
        assert z.shape[-1] == self.quantizer.codebook_embed_size, \
            f"Encoded dimension ({z.shape[-1]}) != quantizer embed size ({self.quantizer.codebook_embed_size})"

        # Quantize
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
        assert not z.isnan().any() and not quantized_indices.isnan().any()

        if use_as_tokenizer:
            return quantized_z, quantized_indices, z

        # Handle multi-codebook indices: MCQ returns [B, num_codebooks, L]
        if quantized_indices.dim() == 3:
            decoder_structure_tokens = quantized_indices[:, 0, :]
        else:
            decoder_structure_tokens = quantized_indices

        # Unproject from MCQ dimension back to decoder dimension
        decoder_input = self.unprojection(quantized_z)

        # Decode and compute losses (reuse parent's loss computation)
        decoded_states = self.decoder.decode(decoder_input, decoder_structure_tokens, attention_mask, sequence_id)

        # Reconstructed proteins - compute RMSD and LDDT if pdb_chain is a WrappedProteinChain
        from protein_chain import WrappedProteinChain
        import numpy as np

        bb_pred = decoded_states["bb_pred"]
        bb_rmsd_list, lddt_list = [], []

        # Check if we have actual chain objects for RMSD/LDDT computation
        # If pdb_chain is a string (sequence only), skip these metrics
        has_chain_objects = pdb_chain is not None and len(pdb_chain) > 0 and hasattr(pdb_chain[0], 'atom37_mask')

        if has_chain_objects:
            for i in range(len(bb_pred)):
                pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[i].detach())
                pdb_chain_recon = pdb_chain_recon[:len(pdb_chain[i])]

                bb_rmsd = pdb_chain_recon.rmsd(pdb_chain[i], only_compute_backbone_rmsd=True)
                lddt = np.array(pdb_chain_recon.lddt_ca(pdb_chain[i]))
                bb_rmsd_list.append(bb_rmsd)
                lddt_list.append(lddt.mean())
        else:
            # Use placeholder values when chain objects are not available
            bb_rmsd_list = [0.0] * len(bb_pred)
            lddt_list = [0.0] * len(bb_pred)

        # Compute reconstruction losses
        coords_recon = decoded_states["bb_pred"]

        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            coords_recon, coords[:, :, :3, :], attention_mask)
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(
            coords_recon, coords[:, :, :3, :], attention_mask)
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(
            decoded_states["pairwise_dist_logits"], coords, attention_mask)
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            decoded_states["pairwise_dir_logits"], coords[:, :, :3, :], attention_mask)
        inverse_folding_loss, inverse_folding_metrics = self.compute_inverse_folding(
            decoded_states["last_hidden_state"], seq_residue_tokens, attention_mask)

        reconstruction_loss = (geom_dist_loss + geom_dir_loss + binned_dist_loss
                              + binned_dir_loss + inverse_folding_loss).mean()
        loss = reconstruction_loss * self.loss_weight["reconstruction_loss_weight"] + partial_loss

        # Contrastive loss (if enabled)
        contrastive_metrics = {}
        if self.use_contrastive:
            chain_ids = [f"{pc.id}_{pc.chain_id}" for pc in pdb_chain]
            texts, ann_mask = self.annotation_loader.get_batch_annotations(chain_ids)
            ann_mask = ann_mask.to(coords.device)
            contrastive_loss, contrastive_metrics = self.clip_loss(
                z, attention_mask, texts, ann_mask
            )
            loss = loss + self.contrastive_loss_weight * contrastive_loss

        # Forward folding path (if enabled)
        forward_folding_metrics = {}
        forward_folding_loss = torch.tensor(0.0, device=coords.device)

        if self.use_forward_folding:
            import torch.nn.functional as F

            z_seq = self.sequence_encoder.encode(seq_residue_tokens, attention_mask, sequence_id)

            with torch.no_grad():
                quantized_indices_seq = self.quantizer.embedding2indices(z_seq)
                quantized_z_seq_detached = self.quantizer.indices2embedding(quantized_indices_seq)

            quantized_z_seq = z_seq + (quantized_z_seq_detached - z_seq).detach()
            commitment_loss_seq = F.mse_loss(z_seq, quantized_z_seq_detached.detach())

            partial_loss_seq = commitment_loss_seq * self.loss_weight.get("commitment_loss_weight", 0.25)
            partial_metrics_seq = {
                "commitment_loss": commitment_loss_seq,
                "vocab_usage": torch.tensor(0.0, device=coords.device),
            }

            if quantized_indices_seq.dim() == 3:
                decoder_structure_tokens_seq = quantized_indices_seq[:, 0, :]
            else:
                decoder_structure_tokens_seq = quantized_indices_seq

            decoded_states_seq = self.decoder.decode(
                quantized_z_seq, decoder_structure_tokens_seq, attention_mask, sequence_id,
                skip_pairwise=self.forward_folding_skip_binned
            )

            coords_pred_from_seq = decoded_states_seq["bb_pred"]

            ff_geom_dist_loss, ff_geom_dist_metrics = self.compute_geometric_distance(
                coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone()
            )
            ff_geom_dir_loss, ff_geom_dir_metrics = self.compute_geometric_direction(
                coords_pred_from_seq, coords[:, :, :3, :].clone(), attention_mask.clone()
            )

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

            forward_folding_loss = (
                ff_geom_dist_loss + ff_geom_dir_loss + ff_binned_dist_loss + ff_binned_dir_loss
            ).mean()
            forward_folding_loss = forward_folding_loss + partial_loss_seq

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

    def get_trainable_parameters(self):
        """Return parameters that should be trained (excludes frozen encoder)."""
        trainable = []

        # Projection layer
        if hasattr(self, 'projection') and not isinstance(self.projection, nn.Identity):
            trainable.extend(self.projection.parameters())

        # Unprojection layer
        if hasattr(self, 'unprojection') and not isinstance(self.unprojection, nn.Identity):
            trainable.extend(self.unprojection.parameters())

        # Quantizer
        trainable.extend(self.quantizer.parameters())

        # Decoder
        trainable.extend(self.decoder.parameters())

        # Inverse folding head
        trainable.extend(self.inverse_folding_head.parameters())

        # Contrastive loss components (if present)
        if self.use_contrastive:
            trainable.extend(self.clip_loss.parameters())

        # Forward folding components (if present)
        if self.use_forward_folding:
            trainable.extend(self.sequence_encoder.parameters())

        return trainable
