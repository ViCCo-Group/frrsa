import unittest

from frrsa.helper.predictor_distance import hadamard, euclidian, check_predictor_data, calculate_pair_indices

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

    def test_empty(self):
        with self.assertRaisesRegex(Exception, 'Predictor has to be a matrix'):
            hadamard_prod, first_pair_idx, second_pair_idx = hadamard(np.array([]))

    def test_not_enough_columns(self):
        with self.assertRaisesRegex(Exception, 'Predictor needs at least 2 columns'):
            hadamard_prod, first_pair_idx, second_pair_idx = hadamard(np.array([[1], [2]]))

    def test_no_matrix(self):
        with self.assertRaisesRegex(Exception, 'Predictor has to be a matrix'):
            hadamard_prod, first_pair_idx, second_pair_idx = hadamard(np.array([1]))

class TestEuclidian(unittest.TestCase):
    def test_small_squared(self):
        predictor = np.array([[6,3,2], [3,5,3]])
        expected_distance = np.array([[9,16,1], [4,0,4]])
        distance, first_pair_idx, second_pair_idx = euclidian(predictor, True)
        expected_first_pair_idx = np.array([0,0,1])
        expected_second_pair_idx = np.array([1,2,2])
        assert_allclose(expected_distance, distance)
        assert_allclose(first_pair_idx, expected_first_pair_idx)
        assert_allclose(second_pair_idx, expected_second_pair_idx)

    def test_small_not_squared(self):
        predictor = np.array([[1,3], [3,10]])
        expected_distance = np.array([[2], [7]])
        distance, first_pair_idx, second_pair_idx = euclidian(predictor, False)
        expected_first_pair_idx = np.array([0])
        expected_second_pair_idx = np.array([1])
        assert_allclose(expected_distance, distance)
        assert_allclose(first_pair_idx, expected_first_pair_idx)
        assert_allclose(second_pair_idx, expected_second_pair_idx)

    def test_empty(self):
        predictor = np.array([])
        with self.assertRaisesRegex(Exception, 'Predictor has to be a matrix'):
            distance, first_pair_idx, second_pair_idx = euclidian(predictor, False)

    def test_not_enough_columns(self):
        predictor = np.array([[1], [2]])
        with self.assertRaisesRegex(Exception, 'Predictor needs at least 2 columns'):
            distance, first_pair_idx, second_pair_idx = euclidian(predictor, False)

    def test_no_matrix(self):
        predictor = np.array([1])
        with self.assertRaisesRegex(Exception, 'Predictor has to be a matrix'):
            distance, first_pair_idx, second_pair_idx = euclidian(predictor, False)

class TestPredictorCheck(unittest.TestCase):
    def test_no_matrix(self):
        predictor = np.array([1])
        with self.assertRaisesRegex(Exception, 'Predictor has to be a matrix'):
            check_predictor_data(predictor)

    def test_not_enough_columns(self):
        predictor = np.array([[1], [2]])
        with self.assertRaisesRegex(Exception, 'Predictor needs at least 2 columns'):
            check_predictor_data(predictor)

class TestIndicesCalculation(unittest.TestCase):
    def test_calcualtion(self):
        predictor = np.array([[1,2,6], [3,4,5]])
        first_pair_idx, second_pair_idx = calculate_pair_indices(3)
        expected_first_pair_idx = np.array([0,0,1])
        expected_second_pair_idx = np.array([1,2,2])
        assert_allclose(first_pair_idx, expected_first_pair_idx)
        assert_allclose(second_pair_idx, expected_second_pair_idx)