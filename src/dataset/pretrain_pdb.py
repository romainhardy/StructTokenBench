import os
import time
import json
from tqdm import tqdm
import gc
from glob import glob

import torch
import torch.distributed as dist

from sklearn.model_selection import train_test_split

from Bio import PDB
import biotite
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from dataset.base import BaseDataset
from dataset.cath import CATHLabelMappingDataset
from esm.utils.constants import esm3 as C
import util
from protein_chain import WrappedProteinChain


class PretrainPDBDataset(BaseDataset):
    """
    Though this class inherents from BaseDataset, they are implemented with 
    different logic of processing, especially indicated by __init__(). Functions
    not designed to be used in this class has "assert" warnings.
    """

    SPLIT_NAME = {
        "test": ["CASP14", "CAMEO"]
    }
    SAVE_SPLIT = ["train", "validation", "fold_test", "superfamily_test"]

    # subsample1.5: 7872; subsample5: 24525; subsample10: 48316; all: 480206

    def __init__(self, *args, **kwargs):
        """
        in kwargs:
            data_version: mmcif_files_filtered_subsample1.5 / mmcif_files_filtered_subsample5 
                        / mmcif_files_filtered_subsample10
            split: "train", "valid", or "test"
            py_logger: python logger
        """
        self.data_path = kwargs["data_path"]
        self.data_version = kwargs["data_version"]
        self.truncation_length = kwargs["truncation_length"]
        self.filter_length = kwargs["filter_length"]
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]
        self.structure_pad_token_id = -1
        self.PDB_DATA_DIR = kwargs["pdb_data_dir"]
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.data_name = kwargs["data_name"]
        self.seq_tokenizer = kwargs["seq_tokenizer"]
        
        if self.split in ["train", "validation"]:
            # load pre-processed data
            target_split_file = self.get_target_file_name()
            processed_flag = os.path.exists(target_split_file)
            
            if not processed_flag:
                assert dist.get_world_size() == 1
                self.process_data_from_scratch_and_save(*args, **kwargs)
                exit(0) # only support for first time preprocessing data and then training for the pretraining data
            else:
                self.data = util.load_sharded_data(target_split_file, self.py_logger)
                self.py_logger.info(f"Loading from processed file {target_split_file},"
                                f"structured data of {len(self.data)} entries.")
        else:
            self.py_logger.info(f"Loading all test datasets")

            file_name = os.path.join(
                self.data_path[:self.data_path.rfind("/data/")], 
                f"./data/utility/{self.split.lower()}/processed_structured_esm3_tokenized_sequence"
            )
            self.py_logger.info(f">>> path: {file_name}")
            assert os.path.exists(file_name), "Test data not preprocessed by ESM3 tokenizers, please run code for 'Code Usage Frequency'"
            self.data = torch.load(file_name, map_location="cpu", weights_only=False)

            self._get_coords_from_pdb_chain()
        
        # Dataset sharding will be done in LightningDataModule
    
    def sanity_check(self):
        assert 0, "Not Needed"
    
    def _get_init_cnt_stats(self):
        return {
            "cnt_chain_fail": 0
        }

    def get_pdb_chain_list(self, pdb_id, chain_id_list):
        # Handle AlphaFold Swiss-Prot structures (AF-* naming convention)
        if pdb_id.startswith("AF-"):
            file = os.path.join(self.PDB_DATA_DIR, f"alphafold_swissprot/{pdb_id}.cif")
        else:
            file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
        protein_chain_list = WrappedProteinChain.from_cif_list(file,
                                            chain_id_list=chain_id_list, id=pdb_id)

        return protein_chain_list

    def _get_coords_from_pdb_chain(self, ):

        # get ESM3's input for VQ-VAE's encoder
        new_data = []
        for i in tqdm(range(len(self.data))):
            pdb_chain = self.data[i]["pdb_chain"]
            coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs()
            
            if len(coords[0]) > self.filter_length:
                continue
            new_data.append(self.data[i])
            
            # filter out residues with nan for N, Ca and C coords
            is_coord_nan = coords[0][:, :3, :].isnan().any(dim=-1).any(dim=-1) # [L, 3, 3] -> [L]
            if is_coord_nan.any():
                indices = is_coord_nan.nonzero()[0]
                if len(indices) > 5:
                    raise ValueError
                pdb_chain = pdb_chain[~is_coord_nan.numpy()]
                coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs()
                new_data[-1]["pdb_chain"] = pdb_chain
            
            new_data[-1]["coords"] = coords[0] # [1, L, 37, 3] -> [L, 37, 3]
            new_data[-1]["plddt"] = plddt[0] # [1, L] -> [L]
            new_data[-1]["residue_index"] = residue_index[0] # [1, L] -> [L]

            sequence = pdb_chain.sequence
            # Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/utils/encoding.py#L48
            if "_" in sequence:
                self.py_logger.info("Somehow character - is in protein sequence")
                raise ValueError
            sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
            seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
            seq_ids = torch.tensor(seq_ids, dtype=torch.int64)
            assert len(seq_ids) == len(coords[0])
            new_data[-1]["seq_ids"] = seq_ids # [L]

        self.py_logger.info(f"After pre-processing, get {len(new_data)} entries")

        self.data = new_data

    
    def process_data_from_scratch_and_save(self, *args, seed=42, **kwargs):

        assert dist.get_world_size() == 1, "dataset not preprocessed and splitted, please not to use multi-GPU training"
        
        # load pdb_ids from id_list
        if self.data_version == "mmcif_files_filtered_subsample1.5":
            tmp = "-sampled1.5"
        elif self.data_version == "mmcif_files_filtered_subsample5":
            tmp = "-sampled5"
        elif self.data_version == "mmcif_files_filtered_subsample10":
            tmp = "-sampled10"
        elif self.data_version == "alphafold_swissprot":
            tmp = None  # Use dedicated Swiss-Prot file
        else:
            assert NotImplementedError

        if tmp is not None:
            file = os.path.join(self.data_path, f"filtered_chain_data_cache{tmp}.json")
        else:
            file = os.path.join(self.data_path, "alphafold_swissprot_chain_data.json")
        
        pdb_chain_id_list = list(json.load(open(file, "rb")).keys())

        cnt_stats = self._get_init_cnt_stats()
        
        # group chain_id for the same pdb_id
        pdb_chain_id_group = {}
        for pdb_chain_id in sorted(pdb_chain_id_list):
            # Use rsplit to handle IDs with multiple underscores (e.g., AlphaFold Swiss-Prot)
            pdb_id, chain_id = pdb_chain_id.strip().rsplit("_", 1)
            if pdb_id in pdb_chain_id_group:
                pdb_chain_id_group[pdb_id].append(chain_id)
            else:
                pdb_chain_id_group[pdb_id] = [chain_id]

        new_data = []
        for i, pdb_id in enumerate(tqdm(list(pdb_chain_id_group.keys()))):
            if self.fast_dev_run and len(new_data) >= 500:
                break
            residue_range = [""]
            pdb_chain_list = self.get_pdb_chain_list(pdb_id, pdb_chain_id_group[pdb_id])
            for pdb_chain in pdb_chain_list:
                if pdb_chain is not None:
                    new_data.append({
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "residue_range": residue_range,
                        "pdb_chain": pdb_chain
                    })
                else:
                    cnt_stats["cnt_chain_fail"] += 1
            if i % 500 == 0:
                self.py_logger.info(f">>> Data loading progress {i}, forming {len(new_data)} samples")
            
        print(f"Loaded all structures: {len(new_data)} samples"
                            f"statistics: {cnt_stats}")

        self.data = new_data
        self._get_coords_from_pdb_chain()

        # random split into training and validation
        data_indices = list(range(len(self.data)))
        data_train_indices, data_valid_indices = train_test_split(data_indices, 
                                                test_size=0.1, random_state=seed)

        data_train_indices = set(data_train_indices)
        train_data, valid_data = [], []
        for i in range(len(self.data)):
            if i in data_train_indices:
                train_data.append(self.data[i])
            else:
                valid_data.append(self.data[i])
        
        tmp = "_fastdev" if self.fast_dev_run else ""
        train_file = os.path.join(self.data_path, f"processed_structured_{self.data_version}_train{tmp}")
        self.py_logger.info(f"Save to training file: {train_file}")
        torch.save(train_data, train_file)
        
        valid_file = os.path.join(self.data_path, f"processed_structured_{self.data_version}_validation{tmp}")
        self.py_logger.info(f"Save to validation file: {valid_file}")
        torch.save(valid_data, valid_file)

        self.py_logger.info(f"Save the processed, structured data to disk: \n"
                            f"with training sample:  {len(data_train_indices)}\n"
                            f"with validation sample: {len(data_valid_indices)}")
        
    def calculate_class_weight(self, ):
        assert 0, "Not in use"
    
    def get_target_file_name(self,):
        tmp = "_fastdev" if self.fast_dev_run else ""
        return os.path.join(self.data_path, f"processed_structured_{self.data_version}_{self.split}{tmp}")
    
    def prepare_structure_loading(self):
        assert 0, "Not in use"

    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        coords, residue_index, seq_ids, pdb_chain, plddt = tuple(zip(*batch))

        coords = util.pad_structures(coords,
                        constant_value=torch.inf,
                        truncation_length=self.truncation_length)
        attention_mask = coords[:, :, 0, 0] == torch.inf
        residue_index = util.pad_structures(residue_index, constant_value=0,
                        truncation_length=self.truncation_length,
                        pad_length=coords.shape[1])

        assert C.SEQUENCE_PAD_TOKEN == 1
        seq_ids = util.pad_structures(seq_ids, constant_value=1, # pad_token_id not work anymore, Jan 14
                        truncation_length=self.truncation_length,
                        pad_length=coords.shape[1])

        # Pad pLDDT with 0 (no contribution from padding)
        plddt = util.pad_structures(plddt, constant_value=0,
                        truncation_length=self.truncation_length,
                        pad_length=coords.shape[1])

        return {
            "input_list": (coords, attention_mask, residue_index, seq_ids, pdb_chain, plddt)
        }
    
    def load_all_structures(self, ):
        assert 0, "Not in use"

    def _get_item_residue_tokens(self, index):
        assert 0, "Not in use"
    
    def _get_item_structural_tokens(self, index):
        assert 0, "Not in use"
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        assert 0, "logic for test data not aligned with train and validation"
        file = os.path.join(self.PDB_DATA_DIR, f"{self.data_version}/{pdb_id}.cif")
        return file

    def __getitem__(self, index: int):
        item = self.data[index]
        coords, residue_index, seq_ids = item["coords"], item["residue_index"], item["seq_ids"]
        # Support both "pdb_chain" (original format) and "sequence" (preprocessed format)
        pdb_chain = item.get("pdb_chain", item.get("sequence", None))
        # pLDDT for confidence-weighted loss (default to 1.0 if not available)
        plddt = item.get("plddt", torch.ones(coords.shape[0]))

        return coords, residue_index, seq_ids, pdb_chain, plddt