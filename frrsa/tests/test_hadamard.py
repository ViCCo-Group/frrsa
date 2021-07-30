import unittest

from helper.predictor_distance import hadamard

import numpy as np
from numpy.testing import assert_allclose

class TestHadamard(unittest.TestCase):
    def test_small(self):
        predictor = np.array([[1,2,6], [3,4,5]])
        hadamard_prod, first_pair_idx, second_pair_idx = hadamard(predictor)
        expected_hadamard_prod = np.array([[2,6,12], [12,15,20]])
        expected_first_pair_idx = np.array([0,0,1])
        expected_second_pair_idx = np.array([1,2,2])
        assert_allclose(hadamard_prod, expected_hadamard_prod)
        assert_allclose(first_pair_idx, expected_first_pair_idx)
        assert_allclose(second_pair_idx, expected_second_pair_idx)