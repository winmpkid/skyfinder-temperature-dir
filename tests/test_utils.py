import unittest

import numpy as np
import torch

from src.utils import compute_lds_weights, get_lds_kernel_window, weighted_l1_loss


class LDSUtilitiesTest(unittest.TestCase):
    def test_gaussian_kernel_is_symmetric_and_normalized(self):
        kernel = get_lds_kernel_window(kernel_size=5, sigma=2.0)

        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        np.testing.assert_allclose(kernel, kernel[::-1])

    def test_lds_weights_have_unit_mean_and_upweight_rare_bins(self):
        temperatures = np.array([0.0] * 20 + [10.0], dtype=np.float64)
        weights, info = compute_lds_weights(
            temperatures,
            bin_width=1.0,
            kernel_size=5,
            sigma=2.0,
        )

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[-1]), float(weights[0]))
        self.assertEqual(info["bin_start"], 0.0)

    def test_weighted_l1_loss_matches_manual_calculation(self):
        predictions = torch.tensor([1.0, 5.0])
        targets = torch.tensor([0.0, 2.0])
        weights = torch.tensor([0.5, 1.5])

        loss = weighted_l1_loss(predictions, targets, weights)

        expected = (0.5 * 1.0 + 1.5 * 3.0) / 2.0
        self.assertAlmostEqual(loss.item(), expected)


if __name__ == "__main__":
    unittest.main()
