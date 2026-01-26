"""
Annotation loading utility for protein functional annotations.
"""

import json
import torch
from typing import Optional


class AnnotationLoader:
    """
    Loads and provides access to protein functional annotations for contrastive learning.
    Implements fallback logic: function -> GO terms -> protein_name + keywords.
    """

    def __init__(self, annotation_path: str):
        """
        Load annotations from JSON file.

        Args:
            annotation_path: Path to training_chain_annotations.json
        """
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

        # Build a case-insensitive lookup (annotations use lowercase PDB IDs)
        self.annotations_lower = {k.lower(): v for k, v in self.annotations.items()}

    def _format_go_terms(self, go_terms: dict) -> str:
        """Format GO terms into a readable text description."""
        parts = []

        # Order: molecular function, biological process, cellular component
        if go_terms.get("molecular_function"):
            mf_terms = [t.split(": ", 1)[-1] if ": " in t else t for t in go_terms["molecular_function"]]
            parts.append(f"Molecular function: {'; '.join(mf_terms)}")

        if go_terms.get("biological_process"):
            bp_terms = [t.split(": ", 1)[-1] if ": " in t else t for t in go_terms["biological_process"]]
            parts.append(f"Biological process: {'; '.join(bp_terms)}")

        if go_terms.get("cellular_component"):
            cc_terms = [t.split(": ", 1)[-1] if ": " in t else t for t in go_terms["cellular_component"]]
            parts.append(f"Cellular component: {'; '.join(cc_terms)}")

        return ". ".join(parts)

    def get_function_text(self, pdb_chain_id: str) -> Optional[str]:
        """
        Get functional text description for a protein chain.
        Implements fallback: function -> GO terms -> protein_name + keywords.

        Args:
            pdb_chain_id: Chain identifier in format "pdbid_chainid" (e.g., "6oma_E")

        Returns:
            Text description or None if no annotation found
        """
        # Try exact match first, then lowercase
        ann = self.annotations.get(pdb_chain_id) or self.annotations_lower.get(pdb_chain_id.lower())

        if ann is None:
            return None

        # Priority 1: Function field (rich descriptions, ~81% coverage)
        if ann.get("function") and ann["function"]:
            func_text = ann["function"]
            if isinstance(func_text, list):
                func_text = " ".join(func_text)
            if func_text.strip():
                return func_text.strip()

        # Priority 2: GO terms (98% coverage as fallback)
        go_terms = ann.get("go_terms", {})
        if any(go_terms.get(k) for k in ["molecular_function", "biological_process", "cellular_component"]):
            go_text = self._format_go_terms(go_terms)
            if go_text.strip():
                return go_text

        # Priority 3: Protein name + keywords
        parts = []
        if ann.get("protein_name"):
            parts.append(ann["protein_name"])
        if ann.get("keywords"):
            keywords = ann["keywords"]
            if isinstance(keywords, list):
                parts.append(f"Keywords: {', '.join(keywords)}")

        if parts:
            return ". ".join(parts)

        return None

    def get_batch_annotations(
        self,
        chain_ids: list[str],
    ) -> tuple[list[str], torch.Tensor]:
        """
        Get functional annotations for a batch of protein chains.

        Args:
            chain_ids: List of chain identifiers (e.g., ["6oma_E", "5iqr_t"])

        Returns:
            texts: List of text descriptions (empty string for missing)
            mask: Boolean tensor indicating which samples have annotations
        """
        texts = []
        mask = []

        for chain_id in chain_ids:
            text = self.get_function_text(chain_id)
            if text is not None:
                texts.append(text)
                mask.append(True)
            else:
                texts.append("")  # Placeholder for missing
                mask.append(False)

        return texts, torch.tensor(mask, dtype=torch.bool)
