import os
import time
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import torch.distributed as dist

from protein_chain import WrappedProteinChain
import util
from dataset.base import BaseDataset, convert_chain_id
from dataset.cath import CATHLabelMappingDataset
import json

from tokenizer import *


class CASP14Dataset(BaseDataset):

    PDB_CHAIN_ID_FILE = "casp14/chain_data_cache.json"
    CIF_DATA_DIR = "casp14/data_dir/"

    SPLIT_NAME = {
        "test": ["test"]
    }

    def process_data_from_scratch(self, *args, **kwargs):

        file_name = os.path.join(self.data_path, self.PDB_CHAIN_ID_FILE)
        pdb_chain_id_list = list(json.load(open(file_name, "rb")).keys())

        self.data = []
        for i in range(len(pdb_chain_id_list)):
            pdb_chain_id = pdb_chain_id_list[i]
            pdb_id, chain_id = pdb_chain_id.split("_")
            self.data.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id
            })
        self.py_logger.info(f"Done preprocessing.")        

    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)

    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    

    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"casp14/processed_structured")

    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        input_ids = batch
        input_ids = util.pad_structures(input_ids, 
                        constant_value=self.structure_pad_token_id, 
                        truncation_length=self.truncation_length)
        
        input_mask = input_ids == self.structure_pad_token_id
        
        return {
            "input_list": (input_ids, input_mask),
            "targets": None
        }
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        file = os.path.join(self.data_path, self.CIF_DATA_DIR, f"{pdb_id}.cif")
        return file
    
    def get_pdb_chain(self, pdb_id, chain_id):
        try:
            file = os.path.join(self.data_path, self.CIF_DATA_DIR, f"{pdb_id}.cif")
            protein_chain = WrappedProteinChain.from_cif(file, chain_id=chain_id, id=pdb_id)
        except:
            source = "local cluster"
            print(f"Cannot retrieve from {source} ", pdb_id, chain_id) # NOTE: temperary
            return None
        return protein_chain
    

    def load_structure(self, index, cnt_stats):
        """Given pdb_id, chain_id
        """

        pdb_id = self.data[index]["pdb_id"]
        chain_id = self.data[index]["chain_id"]
        residue_range = [""]
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE
        
        ret = {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain, 
        }
        return ret
    
    def _get_item_structural_tokens(self, index):
        
        item = self.data[index]
        if "token_ids" in item:
            return item["token_ids"]
    
        pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
        pdb_id, chain_id = item["pdb_id"], item["chain_id"]
        pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)
        
        # convert chain_id if necessary because some chain_id needs to 
        # use use_author_field (specified in biotite).
        chain_id, is_changed = convert_chain_id(pdb_path, chain_id)
        
        assert pdb_chain is not None

        # encode protein structure into token_ids
        if isinstance(self.tokenizer, WrappedESM3Tokenizer):
            # chain_id conversion is already automatically dealt with 
            # WrappedProteinChain, and produced pdb_chain
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, (WrappedFoldSeekTokenizer, WrappedAIDOTokenizer, WrappedProTokensTokenizer)):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedMCQTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence)
        else:
            raise NotImplementedError

        # Handle multi-codebook case: MCQ returns [num_codebooks, L], others return [L]
        if isinstance(token_ids, torch.Tensor) and token_ids.dim() == 2 and not self.use_continuous:
            token_len = token_ids.shape[-1]  # Last dimension is sequence length
        else:
            token_len = len(token_ids)
        assert token_len == len(residue_index)

        # select according to residue range constraints for some global tasks
        assert residue_range == [""]    

        # cache the tokens
        self.data[index]["token_ids"] = token_ids.to("cpu")
        self.data[index]["residue_index"] = residue_index
        return token_ids # torch.Tensor
