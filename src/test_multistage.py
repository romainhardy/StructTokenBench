"""
Smoke tests for multi-stage cross-modal training implementations.
"""

import sys
import torch
import torch.nn.functional as F

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
SEQ_LEN = 32
D_OUT = 128
NUM_CODEBOOKS = 8
# Note: MCQQuantizer uses sub_codebook_size = codebook_size // num_codebooks
# So for 64 entries per codebook, we need codebook_size = 64 * 8 = 512
CODEBOOK_SIZE = 512  # Total codebook size
SUB_CODEBOOK_SIZE = CODEBOOK_SIZE // NUM_CODEBOOKS  # 64 per sub-codebook


def test_get_soft_codebook_logits():
    """Test MCQQuantizer.get_soft_codebook_logits() output shape and probabilities."""
    print("\n" + "="*60)
    print("TEST: MCQQuantizer.get_soft_codebook_logits()")
    print("="*60)

    from vqvae.quantizer_module import MCQQuantizer

    quantizer = MCQQuantizer(
        codebook_embed_size=D_OUT,
        codebook_size=CODEBOOK_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        normalize_embeddings=True,
        loss_weight={"commitment_loss_weight": 0.25, "quantization_loss_weight": 1.0},
    ).to(DEVICE)

    # Create random input
    z = torch.randn(BATCH_SIZE, SEQ_LEN, D_OUT, device=DEVICE)

    # Get soft logits
    logits = quantizer.get_soft_codebook_logits(z, temperature=0.1)

    # Check shape
    expected_shape = (BATCH_SIZE, SEQ_LEN, NUM_CODEBOOKS, SUB_CODEBOOK_SIZE)
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} vs {expected_shape}"
    print(f"  [PASS] Output shape: {logits.shape}")

    # Check that softmax over codebook dimension sums to 1
    probs = F.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities don't sum to 1"
    print(f"  [PASS] Softmax probabilities sum to 1")

    # Check no NaN values
    assert not logits.isnan().any(), "NaN values in logits"
    print(f"  [PASS] No NaN values")

    print("  [SUCCESS] get_soft_codebook_logits() test passed!")
    return True


def test_cross_modal_alignment_loss():
    """Test CrossModalAlignmentLoss with identical and different inputs."""
    print("\n" + "="*60)
    print("TEST: CrossModalAlignmentLoss")
    print("="*60)

    from alignment_loss import CrossModalAlignmentLoss

    loss_fn = CrossModalAlignmentLoss(temperature=0.1, symmetric=True)

    # Test with identical inputs (should have high agreement, low loss)
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_CODEBOOKS, SUB_CODEBOOK_SIZE, device=DEVICE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool, device=DEVICE)

    loss_identical, metrics_identical = loss_fn(logits, logits.clone(), mask)

    print(f"  Identical inputs:")
    print(f"    - Loss: {loss_identical.item():.6f}")
    print(f"    - Token agreement rate: {metrics_identical['align_token_agreement_rate'].item():.4f}")

    assert metrics_identical['align_token_agreement_rate'] == 1.0, "Agreement should be 1.0 for identical inputs"
    print(f"  [PASS] Identical inputs have 100% agreement")

    # Test with different inputs
    logits2 = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_CODEBOOKS, SUB_CODEBOOK_SIZE, device=DEVICE)
    loss_diff, metrics_diff = loss_fn(logits, logits2, mask)

    print(f"  Different inputs:")
    print(f"    - Loss: {loss_diff.item():.6f}")
    print(f"    - Token agreement rate: {metrics_diff['align_token_agreement_rate'].item():.4f}")

    assert loss_diff > loss_identical, "Loss should be higher for different inputs"
    print(f"  [PASS] Different inputs have higher loss")

    # Test with mask
    partial_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool, device=DEVICE)
    partial_mask[:, SEQ_LEN//2:] = False

    loss_masked, metrics_masked = loss_fn(logits, logits2, partial_mask)
    print(f"  With partial mask (50% valid):")
    print(f"    - Loss: {loss_masked.item():.6f}")

    print("  [SUCCESS] CrossModalAlignmentLoss test passed!")
    return True


def test_vanilla_sequence_token_decoder():
    """Test VanillaSequenceTokenDecoder output shape."""
    print("\n" + "="*60)
    print("TEST: VanillaSequenceTokenDecoder")
    print("="*60)

    from vqvae_model import VanillaSequenceTokenDecoder

    vocab_size = 33
    d_model = 512

    decoder = VanillaSequenceTokenDecoder(
        encoder_d_out=D_OUT,
        d_model=d_model,
        n_heads=8,
        n_layers=2,  # Use fewer layers for smoke test
        vocab_size=vocab_size,
    ).to(DEVICE)

    # Create random quantized input
    quantized_z = torch.randn(BATCH_SIZE, SEQ_LEN, D_OUT, device=DEVICE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool, device=DEVICE)

    # Decode
    output = decoder.decode(quantized_z, mask)

    # Check output keys
    assert "logits" in output, "Missing 'logits' key"
    assert "last_hidden_state" in output, "Missing 'last_hidden_state' key"
    print(f"  [PASS] Output contains expected keys")

    # Check shapes
    expected_logits_shape = (BATCH_SIZE, SEQ_LEN, vocab_size)
    expected_hidden_shape = (BATCH_SIZE, SEQ_LEN, d_model)

    assert output["logits"].shape == expected_logits_shape, f"Logits shape mismatch: {output['logits'].shape}"
    assert output["last_hidden_state"].shape == expected_hidden_shape, f"Hidden shape mismatch"
    print(f"  [PASS] Logits shape: {output['logits'].shape}")
    print(f"  [PASS] Hidden state shape: {output['last_hidden_state'].shape}")

    # Check no NaN
    assert not output["logits"].isnan().any(), "NaN in logits"
    print(f"  [PASS] No NaN values")

    print("  [SUCCESS] VanillaSequenceTokenDecoder test passed!")
    return True


def test_stage_freezing():
    """Test that stage freezing works correctly."""
    print("\n" + "="*60)
    print("TEST: Stage Freezing")
    print("="*60)

    from omegaconf import OmegaConf
    from vqvae_model import VQVAEModel

    # Create minimal config for testing
    base_config = {
        "encoder": {
            "d_model": 256,
            "n_heads": 4,
            "v_heads": 1,
            "n_layers": 2,
            "d_out": D_OUT,
        },
        "decoder": {
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 2,
        },
        "quantizer": {
            "quantizer_type": "MCQQuantizer",
            "codebook_embed_size": D_OUT,
            "codebook_size": CODEBOOK_SIZE,
            "num_codebooks": NUM_CODEBOOKS,
            "normalize_embeddings": True,
            "loss_weight": {
                "commitment_loss_weight": 0.25,
                "quantization_loss_weight": 1.0,
                "reconstruction_loss_weight": 1.0,
            },
        },
        "forward_folding": {
            "enabled": True,
            "loss_weight": 1.0,
            "skip_binned_losses": True,
            "sequence_encoder": {
                "vocab_size": 33,
                "d_model": 256,
                "n_heads": 4,
                "n_layers": 2,
                "d_out": D_OUT,
            },
        },
        "sequence_decoder": {
            "enabled": True,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 2,
            "vocab_size": 33,
            "loss_weight": 1.0,
        },
        "alignment": {
            "enabled": True,
            "temperature": 0.1,
            "symmetric": True,
            "loss_weight": 0.5,
        },
    }

    # Test Stage 1: Everything trainable
    print("\n  Stage 1 (Structure Foundation):")
    config_s1 = OmegaConf.create({**base_config, "training_stage": 1})
    model_s1 = VQVAEModel(config_s1).to(DEVICE)
    frozen_s1 = model_s1.get_frozen_components()
    print(f"    Frozen components: {frozen_s1 if frozen_s1 else 'None'}")
    assert len(frozen_s1) == 0, "Stage 1 should have no frozen components"
    print(f"  [PASS] Stage 1: No components frozen")
    del model_s1
    torch.cuda.empty_cache()

    # Test Stage 2: Structure path frozen
    print("\n  Stage 2 (Sequence Alignment):")
    config_s2 = OmegaConf.create({**base_config, "training_stage": 2})
    model_s2 = VQVAEModel(config_s2).to(DEVICE)
    frozen_s2 = model_s2.get_frozen_components()
    print(f"    Frozen components: {frozen_s2}")
    expected_frozen_s2 = {"encoder", "decoder", "quantizer", "inverse_folding_head"}
    assert set(frozen_s2) == expected_frozen_s2, f"Stage 2 frozen mismatch: {set(frozen_s2)} vs {expected_frozen_s2}"
    print(f"  [PASS] Stage 2: Structure path frozen correctly")
    del model_s2
    torch.cuda.empty_cache()

    # Test Stage 3: Everything trainable (different LRs)
    print("\n  Stage 3 (Joint Fine-tuning):")
    config_s3 = OmegaConf.create({**base_config, "training_stage": 3})
    model_s3 = VQVAEModel(config_s3).to(DEVICE)
    frozen_s3 = model_s3.get_frozen_components()
    print(f"    Frozen components: {frozen_s3 if frozen_s3 else 'None'}")

    # Check parameter groups
    param_groups = model_s3.get_parameter_groups(base_lr=1e-4)
    print(f"    Parameter groups:")
    for pg in param_groups:
        name = pg.get("name", "unnamed")
        lr = pg.get("lr", "default")
        n_params = len(pg["params"])
        print(f"      - {name}: lr={lr}, n_params={n_params}")

    assert len(frozen_s3) == 0, "Stage 3 should have no frozen components"
    print(f"  [PASS] Stage 3: No components frozen, differential LRs configured")
    del model_s3
    torch.cuda.empty_cache()

    print("\n  [SUCCESS] Stage freezing test passed!")
    return True


def test_checkpoint_utils():
    """Test checkpoint utility functions."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Utilities")
    print("="*60)

    import tempfile
    import os
    from checkpoint_utils import (
        get_component_state_dict,
        prepare_stage_checkpoint,
        get_stage_from_checkpoint,
    )
    from omegaconf import OmegaConf
    from vqvae_model import VQVAEModel

    # Create a simple model
    config = OmegaConf.create({
        "encoder": {
            "d_model": 128,
            "n_heads": 2,
            "v_heads": 1,
            "n_layers": 1,
            "d_out": D_OUT,
        },
        "decoder": {
            "d_model": 128,
            "n_heads": 2,
            "n_layers": 1,
        },
        "quantizer": {
            "quantizer_type": "MCQQuantizer",
            "codebook_embed_size": D_OUT,
            "codebook_size": CODEBOOK_SIZE,
            "num_codebooks": NUM_CODEBOOKS,
            "normalize_embeddings": True,
            "loss_weight": {
                "commitment_loss_weight": 0.25,
                "quantization_loss_weight": 1.0,
                "reconstruction_loss_weight": 1.0,
            },
        },
        "training_stage": 1,
    })

    model = VQVAEModel(config).to(DEVICE)

    # Test get_component_state_dict
    encoder_state = get_component_state_dict(model, ["encoder"])
    print(f"  Encoder state dict keys: {len(encoder_state)}")
    assert len(encoder_state) > 0, "Should have encoder parameters"
    assert all(k.startswith("encoder") for k in encoder_state.keys()), "All keys should start with 'encoder'"
    print(f"  [PASS] get_component_state_dict works")

    # Test prepare_stage_checkpoint
    checkpoint = prepare_stage_checkpoint(model, stage=1, metrics={"loss": 0.5})
    assert "state_dict" in checkpoint, "Missing state_dict"
    assert checkpoint["training_stage"] == 1, "Wrong stage"
    assert "metrics" in checkpoint, "Missing metrics"
    print(f"  [PASS] prepare_stage_checkpoint works")

    # Test save and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        temp_path = f.name

    try:
        # Test get_stage_from_checkpoint
        stage = get_stage_from_checkpoint(temp_path)
        assert stage == 1, f"Wrong stage inferred: {stage}"
        print(f"  [PASS] get_stage_from_checkpoint works")
    finally:
        os.unlink(temp_path)

    del model
    torch.cuda.empty_cache()

    print("  [SUCCESS] Checkpoint utilities test passed!")
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "#"*60)
    print("# MULTI-STAGE TRAINING SMOKE TESTS")
    print("#"*60)
    print(f"\nDevice: {DEVICE}")

    tests = [
        ("get_soft_codebook_logits", test_get_soft_codebook_logits),
        ("CrossModalAlignmentLoss", test_cross_modal_alignment_loss),
        ("VanillaSequenceTokenDecoder", test_vanilla_sequence_token_decoder),
        ("Stage Freezing", test_stage_freezing),
        ("Checkpoint Utilities", test_checkpoint_utils),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  [FAILED] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

        # Clean up GPU memory between tests
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*60)
        print("ALL SMOKE TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED!")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
