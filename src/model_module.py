import os
import math
import time
import json
import hydra
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import safetensors

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoConfig, AdamW, EsmModel
from torch.optim import Adam
import torch.nn.functional as F
import deepspeed
from pytorch_lightning.utilities import grad_norm


from esm.models.esm3 import ESM3
from tape import ProteinBertModel
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from util import accuracy, get_optimizer, calculate_regression_metric, calculate_binary_clf_metric, calculate_multiclass_clf_metric
from modeling_util import model_init_fn
from vqvae.quantizer_module import get_codebook_utility


class SequenceClassificationHead(nn.Module):

    def __init__(self, 
        in_dim: int, hid_dim: int, out_dim: int, 
        num_layer: int=1, dropout: float=0.0
    ):
        super().__init__()

        dims = [in_dim] + [hid_dim] * num_layer + [out_dim]
        layer_list = []
        for i in range(num_layer + 1):
            layer_list.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layer:
                layer_list.append(nn.ReLU())
                layer_list.append(nn.Dropout(dropout, inplace=False))
        
        self.classify = nn.Sequential(*layer_list)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        return outputs

class ProceedingBaseModel(nn.Module):

    def inference_feature(self, input_list):

        input_ids, input_mask, seqs = input_list

        # add positional embedding
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:,  :seq_length]
        position_embeddings = self.position_embed(position_ids)

        if self.add_noise is None:
            if not self.sequence_only:
                if not hasattr(self, "tokens_embed"):
                    feature = input_ids
                else:
                    feature = self.tokens_embed(input_ids) # [B, L, hidden_size]

                feature += position_embeddings
                if self.use_sequence:
                    sequence_embeddings = self.sequence_embed(seqs)
                    feature += sequence_embeddings
            else:
                assert self.use_sequence
                sequence_embeddings = self.sequence_embed(seqs)
                feature = sequence_embeddings
                feature += position_embeddings
        else:
            if self.sequence_only:
                feature = self.sequence_embed(seqs)
            else:
                if not hasattr(self, "tokens_embed"):
                    feature = input_ids
                else:
                    feature = self.tokens_embed(input_ids)
            # feature: [B, L, hidden_size]
                
            # replace with noise_embedding randomly
            replace_mask = (torch.rand(feature.shape[0], feature.shape[1], device=feature.device) < self.add_noise) # [bsz, L]
            replace_mask = replace_mask.unsqueeze(-1) # [bsz, L, 1]

            noise_embed_broadcasted = self.noise_embedding.weight.view(1, 1, -1)
            feature = torch.where(replace_mask, noise_embed_broadcasted, feature)
            feature += position_embeddings

        feature = self.tokens_layernorm(feature)
        feature = self.tokens_dropout(feature)

        return input_ids, input_mask, feature



class SequenceClassificationModel(ProceedingBaseModel):

    def __init__(self, 
        model_cfg, codebook_embedding=None,
    ):
        super().__init__()

        pretrained_ckpt_path = model_cfg.pretrained_ckpt_path
        num_labels = model_cfg.num_labels
        dropout = model_cfg.dropout
        num_layer = model_cfg.num_layer
        hidden_size = model_cfg.hidden_size
        num_tokens = model_cfg.num_tokens
        d_model = model_cfg.d_model
        is_global_or_local = model_cfg.is_global_or_local

        self.multi_label = model_cfg.multi_label
        self.regression = model_cfg.regression
        self.num_labels = model_cfg.num_labels

        self.pretrained_ckpt_path = pretrained_ckpt_path
        assert pretrained_ckpt_path == ""

        self.codebook_embedding = codebook_embedding
        self.use_sequence = model_cfg.use_sequence
        self.sequence_only = model_cfg.sequence_only
        self.add_noise = model_cfg.add_noise

        if self.add_noise is not None:
            self.noise_embedding = nn.Embedding(1, d_model)
            if self.use_sequence: # prohibit seq + struct tokens for adding noise (not tested)
                assert self.sequence_only

        if self.sequence_only:
            assert self.use_sequence

        self.entering_step = 0

        # load simple neural layers for benchmarking
        if num_tokens is not None:
            self.tokens_embed = nn.Embedding(num_tokens, d_model)
            self.tokens_embed.weight.requires_grad = False
            self.tokens_embed.weight[:len(self.codebook_embedding)] = self.codebook_embedding
        else:
            pass # use the model encoded continuous representations instead
        
        if self.use_sequence:
            self.sequence_embed = nn.Embedding(26 + 1, d_model)
        
        layer_norm_eps = 1e-12
        self.tokens_layernorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.tokens_dropout = nn.Dropout(dropout)
        
        max_position_embeddings = 700 # bounded by cfg.data.filter_length
        self.position_embed = nn.Embedding(max_position_embeddings, d_model)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        
        self.is_global_or_local = is_global_or_local
        if self.is_global_or_local == "global": # protein-wise multi-class classification
            pass
        elif self.is_global_or_local == "local": # residue-wise binary classification
            assert num_labels == 1
        else:
            raise NotImplementedError
        
        self.d_model = d_model
        self.classify = SequenceClassificationHead(
            d_model, hidden_size, num_labels, num_layer, dropout
        )
    def proceed_global_prediction(self, input_ids, input_mask, feature, targets):

        # averge pool for the last hidden state to get seq-level reprs.
        num_of_tokens = (~input_mask).sum(dim=1, keepdim=True)  # (B, 1) or (B, 1, dim)
        if len(input_ids.shape) == 3:
            assert (num_of_tokens[:, :, :-1] == num_of_tokens[:, :,  :-1]).all()
            num_of_tokens = num_of_tokens[:, :, 0] # (B, 1)

        assert not (num_of_tokens == 0).any()
        keep_index = num_of_tokens.squeeze(1) != 0
        num_of_tokens = num_of_tokens[keep_index] # (B, 1)
        if len(input_mask.shape) == 3:
            assert (input_mask[:, :, :-1] == input_mask[:, :, 1:]).all()
            pooled_hidden_state = (feature * (~input_mask)[:, :, 0:1]).sum(dim=1) # (B, dim)
        else:
            pooled_hidden_state = (feature * (~input_mask).unsqueeze(dim=-1)).sum(dim=1)
        pooled_hidden_state = pooled_hidden_state[keep_index]
        pooled_hidden_state = pooled_hidden_state / num_of_tokens  # (B, hidden_size)
        targets = targets[keep_index]
        
        # transform to binary classification tasks
        if self.multi_label:
            raise NotImplementedError
        else:
            logits = self.classify(pooled_hidden_state, targets)[0]
            ret = (logits, )

            if targets is not None:
                valid_indices = (targets != -100).nonzero().squeeze(-1) # (B, ...)
                logits, targets = logits[valid_indices], targets[valid_indices] # (B', ...)

                loss_fct = nn.CrossEntropyLoss()
                classification_loss = loss_fct(logits, targets)
                metrics = calculate_multiclass_clf_metric(logits, targets)
                loss_and_metrics = (classification_loss, metrics)
                ret = (loss_and_metrics, logits, targets)

        return ret

    def proceed_local_prediction(self, input_ids, input_mask, feature, targets):
        if self.regression:
            # input_ids: (B, L) or (B, L, hidden_dim) for ProteinMPNN; feature: (B, L, hidden_size)
            logits = self.classify(feature)[0] # (B, L, 1)
            ret = (logits, )

            if targets is not None:
                # ignore_index default to NaN and the padded -100
                ignore_index = torch.logical_or(targets.isnan(), (torch.abs(targets - (-100)) < 1e-6))
                logits = logits.squeeze(-1)[~ignore_index] # (M, )
                targets = targets[~ignore_index].float() # (M, )

                loss_fct = nn.MSELoss()
                regression_loss = loss_fct(logits, targets)
                metrics = calculate_regression_metric(logits, targets)
                loss_and_metrics = (regression_loss, metrics)

                ret = (loss_and_metrics, logits, targets)
        else:
        
            logits = self.classify(feature)[0] # [B, L, hidden_size] -> [B, L, 1]
            ret = (logits,)
            
            if targets is not None:
                # ignore_index default to -100
                ignore_index = targets == -100 # [B, L]
                logits = logits.squeeze(-1)[~ignore_index] # [M, ]
                targets = targets[~ignore_index].float() # [M, ]

                # data imbalance issue
                ## per-batch class weight
                pos_weight = targets.shape[-1] / targets.sum() * 0.5
                neg_weight = targets.shape[-1] / (targets.shape[-1] - targets.sum()) * 0.5
                ## overal class weight
                class_weight = torch.ones_like(targets) * neg_weight
                class_weight[targets.long() == 1] = pos_weight
                
                loss_fct = nn.BCEWithLogitsLoss(weight=class_weight)
                classification_loss = loss_fct(logits, targets)
                metrics = calculate_binary_clf_metric(logits, targets)
                
                loss_fct_tmp = nn.BCEWithLogitsLoss()
                metrics["unweighted_loss"] = loss_fct_tmp(logits, targets)
                loss_and_metrics = (classification_loss, metrics)
                metrics["pos_weight"] = pos_weight
                metrics["neg_weight"] = neg_weight
                
                ret = (loss_and_metrics, logits, targets)
        return ret 
    
    def forward(self, input_list, targets=None):

        input_ids, input_mask, feature = self.inference_feature(input_list)

        if self.is_global_or_local == "global":
            assert not self.regression
            ret = self.proceed_global_prediction(input_ids, input_mask, feature, targets)
        elif self.is_global_or_local == "local":
            ret = self.proceed_local_prediction(input_ids, input_mask, feature, targets)
        else:
            raise NotImplementedError
        
        return ret


class ZeroShotCodebookUtilityModel(ProceedingBaseModel):

    def __init__(self, model_cfg, codebook_embedding=None):
        super().__init__()

        d_model = model_cfg.d_model
        
        self.d_model = d_model
        self.codebook_embedding = codebook_embedding # [codebook_size, codebook_embed_size]
    
    def forward(self, input_list, target=None):
        input_ids, input_mask = input_list
        metrics = get_codebook_utility(input_ids[~input_mask], self.codebook_embedding.to(input_ids.device))

        # compression bits
        reduced_bits_ratio = []
        for i in range(len(input_ids)):
            tmp = input_ids[i][~input_mask[i]]
            new_bits = sum([len(str(x.item())) for x in tmp])
            # (for each residue, there are 4 backbone atoms, 
            # 3 numbers for xyz, and on average 6 bytes for each float number)
            old_bits = len(tmp) * 4 * 3 * 6
            reduced_bits_ratio.append(new_bits / old_bits)
    
        metrics["compression_ratio"] = torch.mean(torch.tensor(reduced_bits_ratio, device=input_ids.device))

        loss_and_metrics = (torch.zeros(1, device=input_ids.device), metrics)
        return (loss_and_metrics, input_ids[~input_mask], None)


class ZeroshotProximityModel(ProceedingBaseModel):

    def __init__(self, model_cfg, codebook_embedding=None,):
        super().__init__()

        d_model = model_cfg.d_model

        self.d_model = d_model

        if codebook_embedding is not None:
            # for discretized tokenizers
            self.codebook_embedding = codebook_embedding.detach().cpu()
            embed = self.codebook_embedding
            embed = F.normalize(embed, p=2, dim=-1)
            embed = embed.to(torch.float16)
            sim_score = torch.matmul(embed, embed.T)
            sim_score = sim_score.numpy() * 100
            real_num_tokens = sim_score.shape[0]
            self.real_num_tokens = real_num_tokens
            self.alphabet = Alphabet(list(range(real_num_tokens)))
            self.substitution_matrix = SubstitutionMatrix(self.alphabet, self.alphabet, sim_score)
        else:
            # for continuous tokenizers
            self.codebook_embedding = None

    def forward(self, input_list, targets=None):
        """This could be slow because alignment algorithm is currently on CPU.
        Running on GPU could be possible but it's stophisticated
        """
        
        prot1_input_ids, prot2_input_ids = input_list
        # [B, L1], [B, L2] for discretized tokenizers
        # [B, L1, hidden_dim], [B, L2, hidden_dim] for continuous tokenizers

        bsz = prot1_input_ids.shape[0]
        score_list = []
        for i in range(bsz):
            lst1, lst2 = prot1_input_ids[i], prot2_input_ids[i]
            if self.codebook_embedding is not None:
                lst1 = lst1[lst1 < self.real_num_tokens].cpu().numpy()
                lst2 = lst2[lst2 < self.real_num_tokens].cpu().numpy()
                seq1 = GeneralSequence(self.alphabet, lst1)
                seq2 = GeneralSequence(self.alphabet, lst2)
                align_score = align_optimal(seq1, seq2, self.substitution_matrix)[0].score
                score_list.append(align_score)
            else:
                # lst1: [L1, hidden_dim], lst2: [L2, hidden_dim]
                lst1_embed = F.normalize(lst1, p=2, dim=-1)
                lst2_embed = F.normalize(lst2, p=2, dim=-1)
                sim = torch.matmul(lst1_embed, lst2_embed.T) # [L1, L2]
                sim = sim.detach().cpu().numpy() * 100
                L1, L2 = len(lst1_embed), len(lst2_embed)
                sim_score = np.zeros((L1 + L2, L1 + L2))
                sim_score[:L1, L1:] = sim

                alphabet = Alphabet(list(range(L1 + L2)))
                substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
                seq1 = GeneralSequence(alphabet, np.arange(L1))
                seq2 = GeneralSequence(alphabet, np.arange(L2) + L1)
                align_score = align_optimal(seq1, seq2, substitution_matrix)[0].score
                score_list.append(align_score)

        score_list = torch.tensor(score_list, device=targets.device)
        metrics = calculate_regression_metric(score_list.float(), targets)
        
        loss_and_metrics = (torch.zeros(1, device=targets.device), {k:v for k,v in metrics.items() if k != "r2"})

        return (loss_and_metrics, score_list.float(), targets)



class PlModel(pl.LightningModule):
    """
    Pytorch Lightning wrapper class for model training
    """

    def __init__(
        self,
        model_cfg,
        trainer,
        py_logger,
        optimizer_cfg,
        all_split_names,
        codebook_embedding,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.trainer = trainer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        
        self.all_split_names = all_split_names
        for split in self.all_split_names:
            setattr(self, f"{split}_step_outputs", [])

        self.codebook_embedding = codebook_embedding

    def setup(self, stage: str):
        """
        Set up the module, including model creation
        Args:
            stage: Pytorch Lightning stage train/val/test can be used to induce different
                    behavior only used for inheritance
        """

        # DeepSpeed requires batch size in config; DDP doesn't have this attribute
        if hasattr(self.trainer.strategy, 'config') and self.trainer.strategy.config is not None:
            self.trainer.strategy.config["train_micro_batch_size_per_gpu"] = self.optimizer_cfg.micro_batch_size
        self.model = model_init_fn(self.trainer, self.model_cfg, 
                        codebook_embedding=self.codebook_embedding)
        
        # get time here for first iteration at batch 0
        # logged in on_train_batch_end
        self._last_logged_batch_start_time = time.monotonic()

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_list"], batch["targets"])
        loss, metrics = outputs[0]

        self.log(
            "training_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log time/step
        Args:
            outputs: outputs of train_step, not used, required for hook
            batch: use batch to get input/output sequence length
            batch_idx: batch number, not used required for hook
        """

        if batch_idx > 0 and batch_idx % self.trainer.log_every_n_steps == 0:
            # get the time for this iteration
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            # start timeer for the next iteration
            self._last_logged_batch_start_time = time.monotonic()

            time_per_step = elapsed_time / self.trainer.log_every_n_steps

            self.log(
                "sec/step",
                time_per_step,
                on_step=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
            )
        torch.cuda.empty_cache()

    def _valid_or_test_step(self, batch, batch_idx, split="validation"):
        outputs = self.model(batch["input_list"], batch["targets"])
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
        opt_ret = {}
        if batch["targets"] is not None:
            num_sequences = torch.tensor(batch["targets"].shape[0], device=batch["targets"].device)
            opt_ret["num_sequences"] = num_sequences

        logits, targets = outputs[1], outputs[2]
        return {
            f"{split}_loss": loss,
            **log_metrics,
            "logits": logits,
            "targets": targets,
            **opt_ret,
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
        logits, targets = [], []
        for out in outputs:
            for k in out.keys():
                if "logits" in k:
                    logits.append(out[k])
                elif "targets" in k:
                    targets.append(out[k])
                elif k.startswith(split):
                    agg_result[k].append(out[k])
        
        if logits[0] is not None and targets[0] is not None:
            logits, targets = torch.concatenate(logits), torch.concatenate(targets)
        elif logits[0] is not None:
            
            if self.model_cfg.task_goal == "codebook_utilization":
                device = logits[0].device
                utilization_rate = round(np.mean([x.item() for x in agg_result["test_use_ratio"]]), 6)
                perplexity = round(np.mean([x.item() for x in agg_result["test_perplexity_normalized"]]), 6)
                entropy = round(np.mean([x.item() for x in agg_result["test_entropy_normalized"]]), 6)
                agg_result["UR"] = [torch.tensor(utilization_rate, device=device)]
                agg_result["perplexity"] = [torch.tensor(perplexity, device=device)]
                agg_result["entropy"] = [torch.tensor(entropy, device=device)]

            elif self.model_cfg.task_goal == "codebook_diversity":
                
                tk_name = self.trainer.datamodule.tokenizer_name.replace("Wrapped", "").replace("Tokenizer", "").lower()
                if tk_name == "ourpretrained":
                    tk_name += "_" + self.trainer.datamodule.tokenizer_kwargs["ckpt_name"]
                data_name = self.trainer.datamodule.data_args["data_name"].replace("Dataset", "").lower()

                # save codebook
                codevec = self.codebook_embedding

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_codebook_embedding")
                os.makedirs(dir_name, exist_ok=True)
                codebook_file = os.path.join(dir_name, f"codebook_{tk_name}_{data_name}")
                print("Save codebook embeddings to: ", codebook_file)
                torch.save(codevec, codebook_file)

                # save codebook pairwise similarities
                sim_cos_score = F.cosine_similarity(codevec.cpu().unsqueeze(1), codevec.cpu().unsqueeze(0), dim=-1)

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_simscore_dist")
                os.makedirs(dir_name, exist_ok=True)
                similarity_file = os.path.join(dir_name, f"simscore_cos_{tk_name}_{data_name}")
                torch.save(sim_cos_score, similarity_file)
                print("Save codebook similarities to: ", similarity_file)

                # save codebook pairwise similarities weighted by token usage frequency
                input_ids = torch.concatenate(logits)
                def transform_sim(sims):
                    sims = sims.cpu().numpy()
                    index_count = torch.bincount(input_ids, minlength=len(sims))
                    return (index_count, sims)
                used_sim_cos_score = transform_sim(sim_cos_score)

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_simscore_used_dist")
                os.makedirs(dir_name, exist_ok=True)
                used_similarity_file = os.path.join(dir_name, f"simscore_cos_{tk_name}_{data_name}")
                torch.save(used_sim_cos_score, used_similarity_file)
                print("Save codebook similarities weighted by token frequency to: ", used_similarity_file)

                exit(0)
            else:
                raise NotImplementedError
        
        for k in agg_result.keys():
            agg_result[k] = torch.stack(agg_result[k]).mean()

            # recalculate for regression metrics
            if "spearmanr" in k or "pearsonr" in k or "r2" in k:
                tmp = calculate_regression_metric(logits, targets)
                agg_result[k] = tmp[k.split("_")[-1]]

        self.log_dict(
            agg_result,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
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
            norms,
            prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
        )


    def configure_optimizers(self):

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        # filter out bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # filter out layernorm with a variety of spellings
        decay_parameters = [
            name for name in decay_parameters if "layer_norm" not in name
        ]
        decay_parameters = [
            name for name in decay_parameters if "layernorm" not in name
        ]

        params_decay = [
            p
            for n, p in self.model.named_parameters()
            if any(nd in n for nd in decay_parameters)
        ]
        params_nodecay = [
            p
            for n, p in self.model.named_parameters()
            if not any(nd in n for nd in decay_parameters)
        ]

        param_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.optimizer.weight_decay,
            },
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        optimizer = get_optimizer(param_groups, self.optimizer_cfg.optimizer)

        scheduler = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=optimizer)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "validation_loss",
                }
            ],
        )
