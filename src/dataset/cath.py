import os

import pandas as pd

from biotite.sequence.align import align_multiple, SubstitutionMatrix
from biotite.sequence import ProteinSequence

class CATHLabelMappingDataset():
    """
    C.A.T.H.S.O.L.I.D

    C - Class
    A - Architecture
    T - Topology
    H - Homologous Superfamily
    S - Sequence Family (S35)
    O - Orthogous Seqeuce Family (S60)
    L - 'Like' Sequence Family (S95)
    I - Identical (S100)
    D - Domain (S100 count)

    Release CATH 4.3.0
    """

    CATH_STRUCTURAL_CLASSIFICATION_FILE = "./cath-classification-data/cath-domain-list.txt"
    CATH_STRUCTURAL_CLASSIFICATION_FIELDS = [
        "cath_domain", "class_label", "architecture_label",
        "fold_label", "superfamily_label", "S35_cluster_label",
        "S60_cluster_label", "S95_cluster_label", "S100_cluster_label",
        "S100_seq_count", "domain_length", "resolution"
    ]
    
    CATH_SEQUENCE_FILE = "./sequence-data/cath-domain-seqs.fa"

    def __init__(self, *args, **kwargs):
        self.data_path = kwargs["data_path"]


        # load cath_domain and its annotations
        file = os.path.join(self.data_path, self.CATH_STRUCTURAL_CLASSIFICATION_FILE)
        cath_annot = pd.read_csv(file, comment="#", header=None, sep=r'\s+')
        cath_annot.columns = self.CATH_STRUCTURAL_CLASSIFICATION_FIELDS
        cath_annot["pdb_id"] = cath_annot["cath_domain"].apply(lambda x: x[:4])
        cath_annot["pdb_id_chain_id"] = cath_annot["cath_domain"].apply(lambda x: x[:5])
        cath_domain_list = cath_annot["cath_domain"]
        cath_domain_indexing = {x: i for i,x in enumerate(cath_domain_list)}

        # load cath_domain, residue_range and sequence
        file = os.path.join(self.data_path, self.CATH_SEQUENCE_FILE)
        seq_data, rr_data = [0] * len(cath_annot), [0] * len(cath_annot)
        with open(file, "r") as fin:
            for l in fin:
                if l.startswith(">"):
                    l = l.strip().split("|")[-1]
                    cath_domain, residue_range = l.split("/")
                    assert rr_data[cath_domain_indexing[cath_domain]] == 0
                    rr_data[cath_domain_indexing[cath_domain]] = residue_range
                else:
                    assert seq_data[cath_domain_indexing[cath_domain]] == 0
                    seq_data[cath_domain_indexing[cath_domain]] = l.strip()
        
        # assign sequence and residue_range
        cath_annot["seq"] = seq_data
        cath_annot["residue_range"] = rr_data

        self.cath_annot = cath_annot

        # NOTE: the same pdb protein structure could have different domains 
        # corresponding to different fold and superfamily labels

    def retrieve_labels(self, pdb_id, chain_id, ref_seq):
        """Retrieve fold and superfamily labels for data splitting
        `chain_id` could be "None"
        """

        if chain_id == "None":
            indices = self.cath_annot["pdb_id"] == pdb_id
        else:
            key = pdb_id + chain_id
            indices = self.cath_annot["pdb_id_chain_id"] == key
        indices = indices.values.nonzero()[0]
        if len(indices) == 0:
            return None
        
        if ref_seq == "": # only for InterPro
            all_same_flag = True
            tmp = self.cath_annot.iloc[indices]["seq"].values
            for i in range(len(tmp) - 1):
                if tmp[i] != tmp[i+1]:
                    all_same_flag = False

            if all_same_flag:
                matched_idx = indices[0]
                tmp = self.cath_annot.iloc[matched_idx]
                if chain_id == "None":
                    chain_id = tmp["pdb_id_chain_id"][-1]

                return tmp["fold_label"], tmp["superfamily_label"], chain_id
            else:

                return None

        # gather sequences
        query_seqs = [ProteinSequence(ref_seq)]
        for idx in indices:
            tmp_seq = self.cath_annot.iloc[idx]["seq"]
            query_seqs.append(ProteinSequence(tmp_seq))
        
        # multi-sequence alignment
        matrix = SubstitutionMatrix.std_protein_matrix()
        try:
            alignment, order, tree, distances = align_multiple(query_seqs, matrix)
        except:
            print("No alignment")
            return None

        # select the most similar entries in CATH to annotate BioLIP2's data
        matched_idx = indices[distances[0, 1:].argmin()]
        
        tmp = self.cath_annot.iloc[matched_idx]
        if chain_id == "None":
            chain_id = tmp["pdb_id_chain_id"][-1]

        return tmp["fold_label"], tmp["superfamily_label"], chain_id
    
        