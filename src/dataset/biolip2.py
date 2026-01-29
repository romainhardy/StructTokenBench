import os
from tqdm import tqdm
from collections import Counter

import pandas as pd
import torch
import torch.distributed as dist

from dataset.base import BaseDataset
from dataset.cath import CATHLabelMappingDataset

class BioLIP2FunctionDataset(BaseDataset):

    SEQ_ANNOTATION_FILE = "biolip2/BioLiP_updated_set/BioLiP_nr.txt"
    # fields reference to https://zhanggroup.org/BioLiP/download/readme.txt
    SEQ_ANNOTATION_FIELDS = [
        "pdb_id", "receptor_chain", "resolution", "binding_site_label", 
        "ligand_ccd_id", "ligand_chain", "ligand_serial_number", 
        "binding_site_residues_pdb_numbered", "binding_site_residues_renumbered", 
        "catalytic_site_residues_pdb_numbered", "catalytic_site_residues_renumbered", 
        "EC_label", "GO_label", "binding_affinity_pubmed", "binding_affinity_MOAD", 
        "binging_affinity_PDBBindCN", "binding_affinity_BindingDB", "uniprot_id",
        "pubmed_id", "ligand_residue_sequence_number", "receptor_seq"
    ]

    FULL_FIELD_MAPPING = {
        "binding_label": "binding_site_residues_pdb_numbered",
        "catalytic_label": "catalytic_site_residues_pdb_numbered", 
    }

    ALL_DATA_FILE = "biolip2/biolip2_filtered_all"
    SPLIT_NAME = {
        "test": ["fold_test", "superfamily_test"]
    }
    SAVE_SPLIT = ["train", "validation", "fold_test", "superfamily_test"]
    
    def filter_rare_ligand(self, ):
        """Filter out rare ligands with less than 5 protein entries
        """
        cnt_ligand = Counter([x["ligand_ccd_id"] for x in self.data])
        filter_ccd = set([k for k,v in cnt_ligand.items() if v < 5])
        new_data = [x for x in self.data if x["ligand_ccd_id"] not in filter_ccd]
        self.py_logger.info(f"After filtering rare ligand, {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def filter_missing_labels(self, ):
        data_target_field = self.FULL_FIELD_MAPPING[self.target_field]
        new_data = []
        for x in self.data:
            if x[data_target_field] == "?" or isinstance(x[data_target_field], float):
                continue
            new_data.append(x)

        self.py_logger.info(f"After filtering missing labels for {self.target_field}, {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def extract_useful_features(self, ):
        seq_annot_path = os.path.join(self.data_path, self.SEQ_ANNOTATION_FILE)
        seq_annot = pd.read_csv(seq_annot_path, sep="\t", header=None)
        seq_annot.columns = self.SEQ_ANNOTATION_FIELDS

        # transform to data
        self.data = []
        for i in range(len(seq_annot)):
            tmp = {}
            for k in seq_annot.columns:
                tmp[k] = seq_annot.iloc[i][k]
            self.data.append(tmp)

    def process_data_from_scratch(self, *args, **kwargs):
        """The function annotation needed:
            local properties:
                "binding_site_residues_pdb_numbered",
                "catalytic_site_residues_pdb_numbered"
        """
        
        assert dist.get_world_size() == 1, "dataset not preprocessed and splitted, please not to use multi-GPU training"

        all_data_file = os.path.join(self.data_path, self.ALL_DATA_FILE)
        if not os.path.exists(all_data_file):
            self.extract_useful_features()
            # step 1: assign structural classification to proteins
            self.associate_with_CATH_labels() # 78242 -> 32373 entries
            # step 2: filter rare ligand
            self.filter_rare_ligand()
            torch.save(self.data, all_data_file)
        else:
            self.data = torch.load(all_data_file, weights_only=False)
        
        # filter out entries without all four target labels
        self.filter_missing_labels()
        
        res = self.splitting_dataset()
    
        # save to disk
        for i, split in enumerate(self.SAVE_SPLIT):
            target_split_file = os.path.join(self.data_path, f"biolip2/{self.target_field}_{split}")
            torch.save(res[i], target_split_file)
            if split == self.split:
                self.data = res[i]

        self.py_logger.info(f"Done preprocessing, splitting and saving.")
    
    def associate_with_CATH_labels(self, ):

        # associate with CATH labels
        cath_data_path = os.path.join(
            self.data_path[:self.data_path.rfind("/data/")],
            "./data/CATH"
        )
        self.cath_database = CATHLabelMappingDataset(data_path=cath_data_path)
        
        for i in tqdm(range(len(self.data))):
            pdb_id = self.data[i]["pdb_id"]
            chain_id = str(self.data[i]["receptor_chain"])
            ref_seq = self.data[i]["receptor_seq"]
            res = self.cath_database.retrieve_labels(pdb_id, chain_id, ref_seq)
            # None: either cannot find PDB and its chain, 
            # or fail to do multi-sequence alignment
            if res is None:
                self.data[i] = None
            else:
                self.data[i]["fold_label"], self.data[i]["superfamily_label"], _ = res

        new_data = [x for x in self.data if x is not None]
        self.py_logger.info(f"After filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data

    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)

    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    
    def _get_init_cnt_stats(self,):
        cnt_stats = {
            "cnt_return_none": 0,
            "cnt_wrong_pos": 0,
            "cnt_wrong_chain_for_residues": 0
        }
        return cnt_stats
    
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"biolip2/processed_structured_{self.target_field}_{self.split}")
    
    def load_structure(self, idx, cnt_stats):
        """Given pdb_id, chain_id
        """

        pdb_id, chain_id, residue_range, pdb = None, None, [""], None

        pdb_id = self.data[idx]["pdb_id"]
        chain_id = self.data[idx]["receptor_chain"]
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE
    
        # get local labels for each residue
        if self.target_field in ["binding_label", "catalytic_label"]:
            data_target_field = self.FULL_FIELD_MAPPING[self.target_field]
            tmp = self.data[idx][data_target_field].split(" ")
            residues, indices = [x[0] for x in tmp], [x[1:] for x in tmp]
            
            local_label = [0] * len(pdb_chain)
            for rc, ri in zip(residues, indices):
                if ri[-1].isdigit():
                    ri, radd = eval(ri), None
                else:
                    ri, radd = eval(ri[:-1]), ri[-1]
                    radd = ord(radd) - ord("A")
                    
                pos = (pdb_chain.residue_index == ri).nonzero()[0]
                if len(pos) == 0:
                    cnt_stats["cnt_wrong_pos"] += 1
                    continue
                if len(pos) == 1:
                    try:
                        assert pdb_chain.sequence[pos[0]] == rc
                    except:
                        cnt_stats["cnt_wrong_chain_for_residues"] += 1
                    local_label[pos[0]] = 1
                else:
                    for j in range(len(pos)):
                        if pdb_chain.sequence[pos[j]] == rc:
                            break
                    try:
                        assert pdb_chain.sequence[pos[j]] == rc
                    except:
                        cnt_stats["cnt_wrong_chain_for_residues"] += 1
                        raise ValueError
                    local_label[pos[j]] = 1
        else:
            raise NotImplementedError
        
        ret = {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain, 
        }
        ret[self.target_field] = local_label
        return ret
