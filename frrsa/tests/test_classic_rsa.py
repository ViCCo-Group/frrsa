import unittest

from frrsa.helper.classical_RSA import make_RDM, flatten_RDM, correlate_RDMs

import numpy as np
from numpy.testing import assert_allclose

class TestMakeRDM(unittest.TestCase):
    def test_pearson(self):
        activity_matrix = np.array([[0.5, 0.8], [0.6, 0.9]])
        rdm = make_RDM(activity_matrix)
        expected_rdm = np.array([[0.,0.], [0.,0.]])
        assert_allclose(rdm, expected_rdm)

class TestFlattenRDM(unittest.TestCase):
    def test_flatten_single_rdm(self):
        rdm = np.array([[1, 0.5, 0.6], [0.5, 1, 0.7], [0.6, 0.7, 1]])
        flattened_rdm = flatten_RDM(rdm)
        expected_flattened_rdm = np.array([[0.5],
                                           [0.6],
                                           [0.7]])
        assert_allclose(expected_flattened_rdm, flattened_rdm)

    def test_flatten_multiple_rdm(self):
        rdms = np.array([[[1, 2],
                          [0.4, 0.7],
                          [0.5, 0.8]],

                         [[0.4, 0.7],
                          [1, 2],
                          [0.6, 0.9]],

                         [[0.5,  0.8],
                          [0.6, 0.9],
                          [1, 2]]])
        flattened_rdms = flatten_RDM(rdms)
        expected_flattened_rdm = np.array([[0.4, 0.7],
                                           [0.5, 0.8],
                                           [0.6, 0.9]])
        assert_allclose(expected_flattened_rdm, flattened_rdms)

class TestCorrelation(unittest.TestCase):
    def test_pearson(self):
        rdm1 = np.array([0.6, 0.7, 0.8])
        rdm2 = np.array([0.6, 0.7, 0.8])
        pearson, p = correlate_RDMs(rdm1, rdm2, score_type='pearson')
        expected_pearson = 1
        self.assertEqual(pearson, expected_pearson)

    def test_spearman(self):
        rdm1 = np.array([0.6, 0.7, 0.8])
        rdm2 = np.array([0.6, 0.7, 0.8])
        pearson, p = correlate_RDMs(rdm1, rdm2, score_type='spearman')
        expected_pearson = 1
        self.assertEqual(pearson, expected_pearson)