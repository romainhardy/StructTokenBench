import os
import time
import joblib
import pickle
import urllib.request
import functools

import torch
import torch.nn.functional as F
import torch.distributed as dist

import Bio
from Bio import PDB
from biotite.structure.io.pdbx import CIFFile, convert
import numpy as np
from Bio.PDB import Chain
from typing import Literal

# ----- ESM3 Loading ------- #
from esm.models.esm3 import ESM3
from esm.pretrained import (
    ESM3_structure_encoder_v0, 
    ESM3_structure_decoder_v0, 
    ESM3_structure_decoder_v0,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

# ----- MIF Loading ------- #
from sequence_models.pretrained import load_model_and_alphabet as mif_load_model_and_alphabet
from sequence_models.pdb_utils import parse_PDB as mif_parse_PDB
from sequence_models.pdb_utils import process_coords as mif_process_coords


# ----- FoldSeek Loading ------- #
try:
    import mini3di
    from mini3di.utils import last
except ModuleNotFoundError:
    print("[Warining]: mini3di not found")

# ----- ProTokens Loading ------- #
try:
    # load from baselines/ProToken
    from data_process.preprocess import save_pdb_from_aux, protoken_encoder_preprocess, protoken_decoder_preprocess, init_protoken_model
    from data_process.preprocess import protoken_encoder_input_features, protoken_decoder_input_features
    from data_process import residue_constants
except ModuleNotFoundError:
    print("[Warining]: ProToken not found")

# ----- AIDO Loading ------- #

try:
    from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning
    from modelgenerator.structure_tokenizer.datasets.protein_dataset import ProteinDataset
    from modelgenerator.structure_tokenizer.datasets.protein import Protein
except ModuleNotFoundError:
    print("[Warining]: AIDO.st not found")

# ----- Cheap Loading ------- #
try:
    # load from Cheap
    from cheap_proteins.src.cheap.pretrained import (
        CHEAP_shorten_1_dim_64
    )
except ModuleNotFoundError:
    print("[Warining]: cheap not found")
    pass


# ----- ProteinMPNN Loading ------- #
try:
    from baselines.protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB_biounits, parse_PDB
    from baselines.protein_mpnn_cif_parser import parse_cif_pdb_biounits
except ModuleNotFoundError:
    print("[Warining]: ProteinMPNN not found")


from vqvae_model import VQVAEModel


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# Reside Name Mapping:
# {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASX": "B", "CYS": "C", "GLN": "Q", 
# "GLU": "E", "GLX": "Z", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", 
# "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", 
# "VAL": "V", "KCX": "K"}

# All tokenizers should be registered here
# discretized tokenizers can produce both discretized and continuous strutcural reprs.
# while continuous tokenizers can ONLY produce continuous structural reprs.
ALL_TOKENIZER_TYPE = {
    "discretized": [
        "WrappedESM3Tokenizer",
        "WrappedFoldSeekTokenizer",
        "WrappedProTokensTokenizer",
        "WrappedOurPretrainedTokenizer",
        "WrappedAIDOTokenizer",
    ],
    "continuous": [
        "WrappedMIFTokenizer",
        "WrappedProteinMPNNTokenizer",
        "WrappedCheapS1D64Tokenizer"
    ]
}


class WrappedFoldSeekTokenizer():

    FOLDSEEK_STRUC_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
    # reference to https://github.com/althonos/mini3di/blob/faeff98f8c411224fadb73e3878ef6c8ceefa887/mini3di/encoder.py#L27

    def __init__(self, device:  torch.device | str = "cpu"):
        self.device = device

        self.token2id = {s:i for i, s in enumerate(self.FOLDSEEK_STRUC_VOCAB)}
        self.pad_token_id = len(self.FOLDSEEK_STRUC_VOCAB)

        # third party implementation: https://github.com/althonos/mini3di/
        self.tokenizer_encoder = mini3di.Encoder() # based on numpy, not torch tensors
    
    def get_num_tokens(self):
        return len(self.FOLDSEEK_STRUC_VOCAB) + 1 # additional PAD token

    def get_codebook_embedding(self,):        
        return torch.tensor(self.tokenizer_encoder._CENTROIDS) # (20, 2)

    def _encode_structure_mini3di(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        if pdb_path.endswith(".pdb"):
            parser = PDB.PDBParser(QUIET=True)
        else:
            parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("test", pdb_path)
        chain = structure[0][chain_id]
        if not use_continuous:
            states = self.tokenizer_encoder.encode_chain(chain)
            seq_mini3di = self.tokenizer_encoder.build_sequence(states)
        else:
            seq_mini3di = self.hijack_continuous_reprs(chain)
        residues = [residue for residue in chain.get_residues() if "CA" in residue] # aligned with FoldSeek's internal implementation
        residue_index = np.array([_.get_id()[1] for _ in residues])

        seqs = [Bio.PDB.Polypeptide.three_to_index(_.resname) 
                if Bio.PDB.Polypeptide.is_aa(_.resname, standard=True) else 20 for _ in residues] # 20 for unknown AA

        return seq_mini3di, residue_index, seqs
    
    def hijack_continuous_reprs(self, 
        chain: Chain,
        ca_residue: bool = True,
        disordered_atom: Literal["best", "last"] = "best",
    ):
        """
        Adapted from https://github.com/althonos/mini3di/blob/5bc2fb0257e8d743326f74615ee2c1820c66e7c1/mini3di/encoder.py#L60
        """
        # extract residues
        if ca_residue:
            residues = [residue for residue in chain.get_residues() if "CA" in residue]
        else:
            residues = list(chain.get_residues())
        # extract atom coordinates
        r = len(residues)
        ca = np.array(np.nan, dtype=np.float32).repeat(3 * r).reshape(r, 3)
        cb = ca.copy()
        n = ca.copy()
        c = ca.copy()
        for i, residue in enumerate(residues):
            for atom in residue.get_atoms():
                if atom.is_disordered() and disordered_atom == "last":
                    atom = last(atom)
                if atom.get_name() == "CA":
                    ca[i, :] = atom.coord
                elif atom.get_name() == "N":
                    n[i, :] = atom.coord
                elif atom.get_name() == "C":
                    c[i, :] = atom.coord
                elif atom.get_name() == "CB" or atom.get_name() == "CB A":
                    cb[i, :] = atom.coord
        # encode coordinates
        descriptors = self.tokenizer_encoder.feature_encoder.encode_atoms(ca, cb, n, c)
        
        # the last layer is for discretization
        reprs = functools.reduce(lambda x, f: f(x), self.tokenizer_encoder.vae_encoder.layers[:-1], descriptors.data)

        return reprs

    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        assert use_sequence
        
        seq_mini3di, residue_index, seqs = self._encode_structure_mini3di(pdb_path, chain_id, use_continuous)
        if not use_continuous:
            structural_tokens = torch.LongTensor([self.token2id[x] for x in seq_mini3di])
        else:
            structural_tokens = torch.tensor(seq_mini3di, device=self.device)

        # sometimes, seq_mini3di does match the length of pdb_chain from ESM3
        # because mini3di filter residues without CA

        return structural_tokens, residue_index, seqs
    
    
class WrappedProTokensTokenizer():

    """
    Adapted from https://colab.research.google.com/drive/15bBbfa7WigruoME089cSfE242K1MvRGz
    """

    def __init__(self, device=None):
        self.device = device

        dir_name =  os.path.join(os.path.dirname(__file__), "baselines/ProToken")
        self.tokenizer_encoder_1 = init_protoken_model(512, dir_name)
        self.tokenizer_encoder_2 = init_protoken_model(1024, dir_name)

        codebook_file_name = os.path.join(dir_name, "ProToken_Code_Book.pkl")
        self.codebook_embedding = pickle.load(open(codebook_file_name, "rb"))
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return torch.tensor(self.codebook_embedding) # (512, 32)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        assert use_sequence
        encoder_inputs, encoder_aux, seq_len = protoken_encoder_preprocess(pdb_path, task_mode="single", chain_id=chain_id)

        if seq_len <= 512:
            encoder_results = self.tokenizer_encoder_1.encoder(*encoder_inputs)
        elif seq_len <= 1024:
            encoder_results = self.tokenizer_encoder_2.encoder(*encoder_inputs)
        else:
            raise NotImplementedError

        structural_tokens, residue_index, seqs = [], [], []
        for p in range(encoder_aux['seq_mask'].shape[0]):
            if (encoder_aux['seq_mask'][p] 
            and encoder_aux["aatype"][p] < len(residue_constants.restype_order)): # < 20, may be unknown residues like HOH
                structural_tokens.append(encoder_results["protoken_index"][p])
                residue_index.append(encoder_aux["residue_index"][p])
                seqs.append(encoder_aux["aatype"][p])
        
        # tf.Tensor -> np.array -> torch.Tensor
        structural_tokens = torch.tensor(np.asarray(structural_tokens), device=self.device)
        residue_index = np.asarray(residue_index)

        assert (structural_tokens < 512).all()
        return structural_tokens, residue_index, seqs

class WrappedAIDOTokenizer():
    def __init__(self, device=None):
        self.device = device
        self.tokenizer_encoder = EquiformerEncoderLightning("genbio-ai/AIDO.StructureEncoder").to(self.device)
        self.tokenizer_encoder.training = False
        self.codebook_embedding = self.tokenizer_encoder.encoder.codebook.data.cpu()
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return self.codebook_embedding # (512, 384)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        # parse the pdb_file into a Protein object
        if pdb_path.endswith(".pdb"):
            protein = Protein.from_pdb_file_path(pdb_path, chain_id)
        elif pdb_path.endswith(".cif"):
            # 2nd arg (entity_id) will be ignored to ensure the same CIF 
            # parsing logic within this codebase
            protein = Protein.from_cif_file_path(pdb_path, 1, chain_id)
        # Do not implement the cropping logic for simplicity 
        # max_nb_res=1024 by default
        # input_crop = ProteinDataset.protein_to_input_crop(protein)
        protein_input = protein.to_torch_input() # a dict of tensors
        protein_batch = ProteinDataset.collate_fn([protein_input])

        output = self.tokenizer_encoder.forward(
            protein_batch["atom_positions"].to(self.device),
            protein_batch["attention_mask"].to(self.device),
            protein_batch["residue_index"].to(self.device),
        )
        structural_tokens = output["idx"][0]
        residue_index = protein_batch["residue_index"][0]
        seqs = protein_input["aatype"]
        return structural_tokens, residue_index, seqs

class WrappedCheapBaseTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.pad_token_id = 0 # for pipeline compatibility
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None
    
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        sequences = [
            pdb_dict_list[0]["seq"]
        ]
        rep, mask = self.pipeline(sequences) # [bsz=1, L, 64], [bsz=1, L]
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedCheapS1D64Tokenizer(WrappedCheapBaseTokenizer):
    def __init__(self, device):
        self.pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)
        super().__init__(device)

class WrappedOurPretrainedTokenizer():

    def __init__(self, device: torch.device | str = "cpu", model_cfg=None, pretrained_ckpt_path=None, ckpt_name=None):
        self.device = device
        # load
        self.model = VQVAEModel(model_cfg=model_cfg)
        model_states = torch.load(pretrained_ckpt_path, map_location=self.device)["module"]
        new_model_states = {}
        for k,v in model_states.items():
            assert k.startswith("model.")
            new_model_states[k[6:]] = v
        # Use strict=False to handle checkpoints with extra keys (e.g., clip_loss, vocab_usage)
        self.model.load_state_dict(new_model_states, strict=False)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        self.seq_tokenizer = EsmSequenceTokenizer()

        self.ckpt_name = ckpt_name

        # Check if using MCQ (multiple codebooks) or simple VQ (single codebook)
        self.is_mcq = hasattr(self.model.quantizer, 'codebooks')

        # reference: https://github.com/evolutionaryscale/esm/blob/39a3a6cb1e722347947dc375e3f8e2ba80ed8b59/esm/utils/constants/esm3.py#L18C12-L18C35
        self.pad_token_id = self._get_codebook_size() + 3

    def _get_codebook_size(self):
        """Get the total codebook size (handles both MCQ and simple VQ)."""
        if self.is_mcq:
            # MCQ: total size = num_codebooks * sub_codebook_size
            return self.model.quantizer.codebook_size
        else:
            return self.model.quantizer.codebook.weight.shape[0]

    def get_num_tokens(self):
        return self._get_codebook_size() + 5

    def get_codebook_embedding(self,):
        if self.is_mcq:
            # MCQ: concatenate embeddings from all sub-codebooks
            embeddings = []
            for cb in self.model.quantizer.codebooks:
                embeddings.append(cb.weight)
            return torch.cat(embeddings, dim=0)
        else:
            return self.model.quantizer.codebook.weight
    
    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        
        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device) # [1, L, 37, 3], [1, L], [1, L]

        attention_mask = coords[:, :, 0, 0] == torch.inf # [1, L]

        sequence = pdb_chain.sequence
        sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
        
        seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
        seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=self.device)
        assert len(seq_ids) == len(coords[0])
        
        input_list = (coords, attention_mask, residue_index, seq_ids, pdb_chain)
        quantized_reprs, quantized_indices, reprs = self.model(input_list, use_as_tokenizer=True) # [1, L, dim], [1, L], [1, L, dim]

        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence] # total 20 standard AA in Bio

        if use_continuous:
            return reprs.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L, dim], [L]
        else:
            return quantized_indices.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L], [L]


class WrappedESM3Tokenizer():

    
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        
        self.tokenizer_encoder = ESM3_structure_encoder_v0(self.device)
        self.tokenizer_decoder = ESM3_structure_decoder_v0()
        
        # we don't need to attach the whole ESM3 model
        # only get the special token ids from the model
        attached_model = ESM3.from_pretrained("esm3_sm_open_v1")
        self.pad_token_id = attached_model.tokenizers.structure.pad_token_id
        self.bos_token_id = attached_model.tokenizers.structure.bos_token_id
        self.eos_token_id = attached_model.tokenizers.structure.eos_token_id
    
    def get_num_tokens(self):
        # reference from https://github.com/evolutionaryscale/esm/blob/17d48878a9cfad388fdf5ff4d3fe4ea0f0d24839/esm/models/esm3.py#L85
        return self.tokenizer_encoder.codebook.n_codes + 5 # 4096 + 5

    def get_codebook_embedding(self,):
        return self.tokenizer_encoder.codebook.embeddings # (4096, 128)

    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        """Reference from https://github.com/evolutionaryscale/esm/blob/95e3c5be8acda407414810ff3aa7d27dbb6e30d3/esm/utils/encoding.py#L60
        """

        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device)
        #coords: (1, L, 37, 3), plddt: (1, L), residue_index: (1, L)
        
        if not use_continuous:
            _, structure_tokens = self.tokenizer_encoder.encode(coords, residue_index=residue_index) # _, (1, L)
        else:
            structure_tokens = self.hijack_continuous_reprs(coords, residue_index=residue_index)
        coords = torch.squeeze(coords, dim=0)  # (L, 37, 3)  # type: ignore
        plddt = torch.squeeze(plddt, dim=0)  # (L,)  # type: ignore
        structure_tokens = torch.squeeze(structure_tokens, dim=0)  # (L,)  # type: ignore

        assert len(pdb_chain.residue_index) == len(pdb_chain.sequence)

        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence] # total 20 standard AA in Bio
        return structure_tokens, np.array(pdb_chain.residue_index), seqs
    
    def hijack_continuous_reprs(self, coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        """
        Adapted from https://github.com/evolutionaryscale/esm/blob/39a3a6cb1e722347947dc375e3f8e2ba80ed8b59/esm/models/vqvae.py#L301
        """
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.tokenizer_encoder.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.tokenizer_encoder.pre_vq_proj(z)

        return z

class WrappedMIFTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.model, self.mif_collater = mif_load_model_and_alphabet('mif')
        self.pad_token_id = 0 # for pipeline compatibility
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        coords = {
            'N': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"N_chain_{chain_id}"]),
            'CA': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"CA_chain_{chain_id}"]),
            'C': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"C_chain_{chain_id}"])
        }
        dist, omega, theta, phi = mif_process_coords(coords)
        batch = [[pdb_dict_list[0][f"seq_chain_{chain_id}"], torch.tensor(dist, dtype=torch.float),
              torch.tensor(omega, dtype=torch.float),
              torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
        src, nodes, edges, connections, edge_mask = self.mif_collater(batch)
        rep = self.model(src, nodes, edges, connections, edge_mask)

        assert len(pdb_dict_list[0]["seq"]) == len(rep[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedProteinMPNNTokenizer():
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    MODEL_CKPT_BASEURL = "https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/"
    CKPT_CACHE_DIR = os.path.join(SCRIPT_PATH, 'ProteinMPNN')
    CA_ONLY = False

    def __init__(
            self, 
            device: torch.device | str = "cpu", 
            checkpoint_path: str = "vanilla_model_weights/v_48_020.pt"):
        self.device = device
        # init model and load model weights
        local_checkpoint_path = self._download_model_checkpoint(checkpoint_path)
        self._load_protein_mpnn_model(local_checkpoint_path)

        self.pad_token_id = 0 # for pipeline compatibility

    def _download_model_checkpoint(self, checkpoint_path):
        """Download ProteinMPNN checkpoint from GitHub if not locally cached."""
        ckpt_url = self.MODEL_CKPT_BASEURL + checkpoint_path
        cached_checkpoint_path = os.path.join(self.CKPT_CACHE_DIR, checkpoint_path)
        os.makedirs(os.path.dirname(cached_checkpoint_path), exist_ok=True)
        if not os.path.isfile(cached_checkpoint_path):
            urllib.request.urlretrieve(ckpt_url, cached_checkpoint_path)
        return cached_checkpoint_path

    def _load_protein_mpnn_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = ProteinMPNN(
            ca_only=self.CA_ONLY, 
            num_letters=21, 
            node_features=self.HIDDEN_DIM, 
            edge_features=self.HIDDEN_DIM, 
            hidden_dim=self.HIDDEN_DIM, 
            num_encoder_layers=self.NUM_LAYERS, 
            num_decoder_layers=self.NUM_LAYERS, 
            augment_eps=0.0, 
            k_neighbors=checkpoint['num_edges'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device)

    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence
        
        parse_fn = parse_cif_pdb_biounits
        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=self.CA_ONLY,
            parse_fn=parse_fn
            )

        # this function comes from ProteinMPNN:
        X, _, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
            pdb_dict_list, 
            self.device, None, None, None, None, None, None, 
            ca_only=self.CA_ONLY)

        h_V, h_E = self.model.encode(X, mask, residue_idx, chain_encoding_all)
        assert len(pdb_dict_list[0]["ridx"]) == len(h_V[0])
        # h_V: [1, L, hidden_dim]
        # h_E: [1, L, n_edges, hidden_dim] 

        assert len(pdb_dict_list[0]["seq"]) == len(h_V[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio
        return h_V.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, 128], [L]