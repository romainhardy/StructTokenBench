import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import *
from tokenizer import *
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from tqdm import tqdm

def get_tokenizer_device(tokenizer_device) -> torch.device:
    """Get CPU or local GPU
    """
    if tokenizer_device == "cuda":
        gpu_idx = 0
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        device = torch.device(f"{tokenizer_device}:{gpu_idx}")
    elif tokenizer_device == "cpu":
        device = torch.device(tokenizer_device)
    return device

class ProteinDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer_name: str, tokenizer_device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only: bool, 
        precompute_tokens: bool, tokenizer_kwargs: dict,
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer_device = tokenizer_device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only
        self.precompute_tokens = precompute_tokens

        if self.test_only:
            self.all_split_names = []
        else:
            self.all_split_names = ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]
        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {} 

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        pass

    def _setup_tokenizer(self):
        """Initialize the tokenizer on appropriate device. 
        """
        device = get_tokenizer_device(self.tokenizer_device)
        if self.tokenizer_name.startswith("Wrapped"):
            # all Wrapped tokenizers deal with loading logic inside __init__() when built up
            # assume only this type of tokenizer needs to be device aware
            tokenizer = eval(self.tokenizer_name)(device=device, **self.tokenizer_kwargs)
        else:
            raise NotImplementedError
        
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer
    
    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.tokenizer_device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer

    def get_codebook_embedding(self,):
        tokenizer = self.get_tokenizer()
        return tokenizer.get_codebook_embedding()

    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        # need to shard the dataset here:
        assert torch.distributed.is_initialized()
        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        dataset.shard(shard_idx=process_global_rank, num_shards=world_size)

        if self.precompute_tokens:
            # precompute and cache the token ids:
            self.py_logger.info(
                f"Precomputing tokenized ids on {process_global_rank} with world size {world_size}..."
            )
            dataset.cache_all_tokenized()

            # Verify token lengths match sequence lengths (only after precomputing)
            if dataset.data_name not in ["ConformationalSwitchDataset", "CASP14Dataset", "CAMEODataset"]:
                for i in tqdm(range(len(dataset.data))):
                    token_ids = dataset.data[i]["token_ids"]
                    # Handle multi-codebook case (MCQ): token_ids has shape [num_codebooks, L]
                    if isinstance(token_ids, torch.Tensor) and token_ids.dim() == 2:
                        token_len = token_ids.shape[-1]  # Use last dimension for sequence length
                    else:
                        token_len = len(token_ids)
                    assert len(dataset.data[i]["real_seqs"]) == token_len
        return dataset

    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None
        
        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders


class PretrainingDataModule(pl.LightningDataModule):

    def __init__(self, device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only,
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only

        self.all_split_names = []
        if not self.test_only:
            self.all_split_names += ["validation"]
        # Only include test splits if skip_test_validation is not set
        # (test splits require ESM3-tokenized data which may not be available)
        if not getattr(self.data_args, 'skip_test_validation', False):
            self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]

        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {}
    
    def _setup_tokenizer(self):
        device = get_tokenizer_device(self.device)
        tokenizer = EsmSequenceTokenizer()
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer

    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer 
    
    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "seq_tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        # need to shard the dataset here:
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            dataset.shard(shard_idx=process_global_rank, num_shards=world_size)
        return dataset
    
    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None

        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders

    def setup(self, stage=None):
        pass
