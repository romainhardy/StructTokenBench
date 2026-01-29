"""
Evaluation utilities for structured experiment logging.

This module provides functionality to save experiment results in a structured
JSON format for easy comparison across different tokenizers and benchmarks.
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_experiment_results(
    experiment_name: str,
    description: str,
    config: Dict[str, Any],
    tokenizer_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_checkpoint_path: Optional[str] = None,
    output_dir: str = ".",
) -> str:
    """
    Save experiment results to a structured JSON file.

    Args:
        experiment_name: Unique identifier for the experiment
        description: Human-readable description of what was tested
        config: Full hydra config dict (for reference)
        tokenizer_info: Dict with name, checkpoint, num_tokens, d_model
        dataset_info: Dict with name, split sizes, etc.
        hyperparameters: Dict with lr, batch_size, max_steps, seed
        validation_metrics: Dict of validation metric name -> value
        test_metrics: Dict of test metric name -> value
        best_checkpoint_path: Path to the best model checkpoint
        output_dir: Directory to save the results file

    Returns:
        Path to the saved results file
    """
    results = {
        "experiment_name": experiment_name,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "tokenizer": tokenizer_info,
        "dataset": dataset_info,
        "hyperparameters": hyperparameters,
        "best_checkpoint": best_checkpoint_path,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        # Store minimal config info for reference
        "config_snapshot": {
            "model_class": config.get("model", {}).get("class_name"),
            "tokenizer_class": config.get("tokenizer"),
            "use_continuous": config.get("data", {}).get("use_continuous"),
            "use_sequence": config.get("model", {}).get("use_sequence"),
        }
    }

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Also append to a cumulative results file for easy comparison
    cumulative_file = os.path.join(output_dir, "all_results.jsonl")
    with open(cumulative_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    return filepath


def extract_metrics_from_trainer(trainer, prefix: str = "") -> Dict[str, float]:
    """
    Extract metrics from a PyTorch Lightning trainer's callback metrics.

    Args:
        trainer: PyTorch Lightning Trainer instance
        prefix: Filter metrics by prefix (e.g., "test_", "validation_")

    Returns:
        Dict of metric name -> value
    """
    metrics = {}
    callback_metrics = trainer.callback_metrics

    for key, value in callback_metrics.items():
        if prefix and not key.startswith(prefix):
            continue
        # Convert tensor to float if needed
        if hasattr(value, "item"):
            metrics[key] = value.item()
        else:
            metrics[key] = float(value)

    return metrics


def load_results(results_dir: str) -> list:
    """
    Load all results from a results directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of result dicts
    """
    cumulative_file = os.path.join(results_dir, "all_results.jsonl")
    results = []

    if os.path.exists(cumulative_file):
        with open(cumulative_file, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    return results


def print_results_table(results: list, metric_keys: list = None):
    """
    Print a comparison table of results.

    Args:
        results: List of result dicts
        metric_keys: List of metric keys to display (default: test_f1_score, test_auroc)
    """
    if metric_keys is None:
        metric_keys = ["test_f1_score", "test_auroc"]

    # Header
    header = ["Experiment", "Tokenizer"] + metric_keys
    print(" | ".join(f"{h:>20}" for h in header))
    print("-" * (22 * len(header)))

    # Rows
    for r in results:
        row = [
            r["experiment_name"][:20],
            r["tokenizer"].get("name", "unknown")[:20]
        ]
        for key in metric_keys:
            val = r.get("test_metrics", {}).get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        print(" | ".join(f"{v:>20}" for v in row))
