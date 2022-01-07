import unittest

from frrsa.helper.classical_RSA import make_RDM, flatten_RDM, correlate_RDMs

import numpy as np
from numpy.testing import assert_allclose

class TestMakeRDM(unittest.TestCase):
    def test_pearson(self):
        activity_pattern_matrix = np.array([[0.5, 0.8], 
                                            [0.6, 0.9]])
        rdm = make_RDM(activity_pattern_matrix, distance='pearson')
        expected_rdm = np.array([[0.,0.], 
                                 [0.,0.]])
        assert_allclose(rdm, expected_rdm)

class TestFlattenRDM(unittest.TestCase):
    def test_flatten_single_rdm(self):
        rdm = np.array([[1, 0.5, 0.6], 
                        [0.5, 1, 0.7], 
                        [0.6, 0.7, 1]])
        rdv = flatten_RDM(rdm)
        expected_rdv = np.array([[0.5],
                                 [0.6],
                                 [0.7]])
        assert_allclose(expected_rdv, rdv)

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
        rdv = flatten_RDM(rdms)
        expected_rdv = np.array([[0.4, 0.7],
                                 [0.5, 0.8],
                                 [0.6, 0.9]])
        assert_allclose(expected_rdv, rdv)

class TestCorrelateRDMs(unittest.TestCase):
    def test_pearson(self):
        rdv1 = np.array([0.6, 0.7, 0.8])
        rdv2 = np.array([0.6, 0.7, 0.8])
        corr, p = correlate_RDMs(rdv1, rdv2, score_type='pearson')
        expected_corr = 1
        self.assertEqual(corr, expected_corr)

    def test_spearman(self):
        rdv1 = np.array([0.6, 0.7, 0.8])
        rdv2 = np.array([0.6, 0.7, 0.8])
        corr, p = correlate_RDMs(rdv1, rdv2, score_type='spearman')
        expected_corr = 1
        self.assertEqual(corr, expected_corr)