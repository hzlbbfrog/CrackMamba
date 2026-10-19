"""Regression tests for selective recurrence, serpentine geometry and checkpoints."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model tests")
class ModelTests(unittest.TestCase):
    def test_closed_form_recurrence(self):
        import math
        import torch
        from models.selective_scan import reference_scan

        # Softplus(log(exp(1)-1))=1 and exp(-log(2))=0.5.
        u = torch.ones(1, 4, 5)
        delta = torch.zeros_like(u)
        decay = torch.full((4, 1), -math.log(2))
        source = torch.full((1, 2, 1, 5), 2.0)
        readout = torch.full_like(source, 3.0)
        skip = torch.full((4,), 0.1)
        bias = torch.full((4,), math.log(math.expm1(1.0)))
        actual = reference_scan(u, delta, decay, source, readout, skip, bias)
        expected = torch.tensor([12 * (1 - 0.5 ** t) + 0.1 for t in range(1, 6)])
        torch.testing.assert_close(actual, expected[None, None, :].expand(1, 4, 5))

    def test_numpy_bridge_matches_torch_recurrence(self):
        import torch
        from models.selective_scan import reference_scan, _numpy_reference

        torch.manual_seed(7)
        u, delta = torch.randn(1, 8, 7), torch.randn(1, 8, 7)
        decay = -torch.rand(8, 3)
        source, readout = torch.randn(1, 2, 3, 7), torch.randn(1, 2, 3, 7)
        skip, bias = torch.randn(8), torch.randn(8)
        actual = _numpy_reference(u, delta, decay, source, readout, skip, bias)
        expected = reference_scan(u, delta, decay, source, readout, skip, bias)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_all_scan_directions_restore_rectangular_coordinates(self):
        import torch
        from models.crackseu_global import make_weight
        from models.global_scan import SS2D_forward

        # With an identity scan, every restored path must be the original image.
        for branch in (1, 7):
            for flag in (0, 1):
                height, width = (2, 3) if flag == 0 else (3, 2)
                x = torch.arange(12, dtype=torch.float32).reshape(1, 2, height, width)
                weights = [make_weight(1, 1, 2, 3).expand(1, 1, 2, -1),
                           make_weight(1, 1, 3, 2).expand(1, 1, 2, -1)]
                module = SS2D_forward(2, d_state=1, branch=branch)
                with patch("models.global_scan.selective_scan", side_effect=lambda values, *args, **kwargs: values):
                    actual = module(x, weights, flag)
                torch.testing.assert_close(actual, 4 * x)

    def test_published_checkpoint_loads_all_entries_strictly(self):
        import torch
        from models.crackseu_global import load_model

        checkpoint = ROOT / "checkpoints" / "crackmamba_global_deepcrack_epoch15.pth"
        self.assertTrue(checkpoint.is_file())
        torch.set_num_threads(1)
        model = load_model(checkpoint, "cpu", "reference")
        self.assertEqual(len(model.state_dict()), 237)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
