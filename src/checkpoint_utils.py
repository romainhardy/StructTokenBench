"""
Checkpoint utilities for multi-stage training.

Provides functions for:
- Loading checkpoints from previous training stages
- Validating stage transitions
- Extracting component-specific state dicts
"""

import os
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_stage_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    map_location: str = "cpu",
) -> Tuple[Set[str], Set[str]]:
    """
    Load weights from a previous stage checkpoint.

    Uses strict=False by default to allow loading checkpoints that may have
    different components (e.g., loading stage 1 checkpoint into stage 2 model
    which has additional sequence decoder).

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file (.pt, .ckpt, or .safetensors)
        strict: If True, requires all keys to match exactly
        map_location: Device to load checkpoint to

    Returns:
        Tuple of (missing_keys, unexpected_keys) from the load operation

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
        ValueError: If checkpoint format is not supported
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Determine checkpoint format and load
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        except ImportError:
            raise ImportError("safetensors package required for .safetensors files")
    elif checkpoint_path.endswith(".ckpt"):
        # PyTorch Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove "model." prefix if present (Lightning adds this)
            state_dict = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
        else:
            state_dict = checkpoint
    else:
        # Standard PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    # Load state dict
    result = model.load_state_dict(state_dict, strict=strict)

    missing_keys = set(result.missing_keys) if hasattr(result, 'missing_keys') else set()
    unexpected_keys = set(result.unexpected_keys) if hasattr(result, 'unexpected_keys') else set()

    if missing_keys:
        logger.info(f"Missing keys (expected for new components): {len(missing_keys)}")
        for key in sorted(missing_keys)[:10]:  # Show first 10
            logger.debug(f"  - {key}")
        if len(missing_keys) > 10:
            logger.debug(f"  ... and {len(missing_keys) - 10} more")

    if unexpected_keys:
        logger.warning(f"Unexpected keys (may indicate version mismatch): {len(unexpected_keys)}")
        for key in sorted(unexpected_keys)[:10]:
            logger.warning(f"  - {key}")

    return missing_keys, unexpected_keys


def validate_stage_transition(
    current_stage: int,
    checkpoint_stage: Optional[int],
    model: nn.Module,
    missing_keys: Set[str],
) -> bool:
    """
    Validate that a stage transition is valid.

    Checks that:
    1. We're transitioning to a higher stage
    2. Required components from previous stage are present
    3. Missing keys are only for newly added components

    Args:
        current_stage: The stage we're training (1, 2, 3, or 4)
        checkpoint_stage: The stage of the loaded checkpoint (None if starting fresh)
        model: The model being trained
        missing_keys: Keys missing from the loaded checkpoint

    Returns:
        True if transition is valid, False otherwise
    """
    if checkpoint_stage is None:
        # Starting fresh - only valid for stage 1
        if current_stage != 1:
            logger.warning(f"Starting stage {current_stage} without checkpoint. "
                          f"Consider loading from stage {current_stage - 1}.")
            return True  # Allow but warn
        return True

    if checkpoint_stage >= current_stage:
        logger.warning(f"Loading stage {checkpoint_stage} checkpoint for stage {current_stage}. "
                      f"This may not be intended.")
        return True  # Allow but warn

    # Define expected new components for each stage transition
    expected_new_components = {
        2: ["sequence_encoder", "sequence_decoder", "alignment_loss"],
        3: [],  # Stage 3 uses all existing components
        4: ["function_encoder", "function_decoder"],
    }

    if current_stage in expected_new_components:
        expected_prefixes = expected_new_components[current_stage]
        unexpected_missing = set()

        for key in missing_keys:
            is_expected = any(key.startswith(prefix) for prefix in expected_prefixes)
            if not is_expected:
                unexpected_missing.add(key)

        if unexpected_missing:
            logger.warning(f"Unexpected missing keys for stage {current_stage} transition:")
            for key in sorted(unexpected_missing)[:5]:
                logger.warning(f"  - {key}")
            return False

    logger.info(f"Stage transition {checkpoint_stage} -> {current_stage} validated successfully")
    return True


def get_component_state_dict(
    model: nn.Module,
    component_names: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Extract state dict for specific model components.

    Useful for saving only certain parts of the model or for transferring
    weights between models.

    Args:
        model: The model to extract from
        component_names: List of component names (e.g., ["encoder", "decoder"])

    Returns:
        State dict containing only the specified components
    """
    full_state_dict = model.state_dict()
    filtered_state_dict = {}

    for name in component_names:
        prefix = f"{name}."
        for key, value in full_state_dict.items():
            if key.startswith(prefix) or key == name:
                filtered_state_dict[key] = value

    logger.info(f"Extracted {len(filtered_state_dict)} parameters for components: {component_names}")
    return filtered_state_dict


def get_stage_from_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Optional[int]:
    """
    Try to determine the training stage from a checkpoint.

    Looks for stage information in the checkpoint metadata or infers from
    the components present.

    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to load checkpoint to

    Returns:
        Training stage (1, 2, 3, or 4) or None if cannot be determined
    """
    if not os.path.exists(checkpoint_path):
        return None

    try:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            # safetensors doesn't have metadata in the same way
            checkpoint = {"state_dict": state_dict}
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Check for explicit stage in metadata
        if "training_stage" in checkpoint:
            return checkpoint["training_stage"]
        if "hyper_parameters" in checkpoint:
            hp = checkpoint["hyper_parameters"]
            if "training_stage" in hp:
                return hp["training_stage"]

        # Infer from components present
        state_dict = checkpoint.get("state_dict", checkpoint)
        keys = set(state_dict.keys())

        # Check for stage-specific components
        has_function = any(k.startswith("function_encoder") or k.startswith("function_decoder") for k in keys)
        has_sequence = any(k.startswith("sequence_encoder") or k.startswith("sequence_decoder") for k in keys)
        has_alignment = any(k.startswith("alignment_loss") for k in keys)

        if has_function:
            return 4
        elif has_sequence or has_alignment:
            return 2  # Could be 2 or 3, conservatively return 2
        else:
            return 1

    except Exception as e:
        logger.warning(f"Could not determine stage from checkpoint: {e}")
        return None


def prepare_stage_checkpoint(
    model: nn.Module,
    stage: int,
    optimizer=None,
    scheduler=None,
    metrics: Optional[Dict] = None,
) -> Dict:
    """
    Prepare a checkpoint dict with stage information.

    Args:
        model: The model to save
        stage: Current training stage
        optimizer: Optional optimizer to include
        scheduler: Optional scheduler to include
        metrics: Optional metrics dict to include

    Returns:
        Checkpoint dict ready for torch.save()
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "training_stage": stage,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    # Record which components are frozen
    if hasattr(model, 'get_frozen_components'):
        checkpoint["frozen_components"] = model.get_frozen_components()

    return checkpoint


def load_and_freeze_components(
    model: nn.Module,
    checkpoint_path: str,
    components_to_freeze: List[str],
    map_location: str = "cpu",
) -> None:
    """
    Load a checkpoint and freeze specific components.

    Useful for stage transitions where some components should be frozen
    after loading.

    Args:
        model: The model to load into
        checkpoint_path: Path to the checkpoint
        components_to_freeze: List of component names to freeze after loading
        map_location: Device to load checkpoint to
    """
    # Load checkpoint
    load_stage_checkpoint(model, checkpoint_path, strict=False, map_location=map_location)

    # Freeze specified components
    for name in components_to_freeze:
        component = getattr(model, name, None)
        if component is not None and isinstance(component, nn.Module):
            for param in component.parameters():
                param.requires_grad = False
            logger.info(f"Froze component: {name}")
        else:
            logger.warning(f"Component not found or not a module: {name}")
