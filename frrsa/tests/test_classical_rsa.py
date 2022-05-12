#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from frrsa.helper.classical_RSA import make_RDM, flatten_matrix

import numpy as np
from numpy.testing import assert_allclose


class TestMakeRDM(unittest.TestCase):
    def test_pearson(self):
        activity_pattern_matrix = np.array([[0.5, 0.8],
                                            [0.6, 0.9]])
        rdm = make_RDM(activity_pattern_matrix, distance='pearson')
        expected_rdm = np.array([[0., 0.],
                                 [0., 0.]])
        assert_allclose(rdm, expected_rdm)

    def test_sqeuclidean(self):
        activity_pattern_matrix = np.array([[1, 4],
                                            [2, 1]])
        rdm = make_RDM(activity_pattern_matrix, distance='sqeuclidean')
        expected_rdm = np.array([[0., 10.],
                                 [10., 0.]])
        assert_allclose(rdm, expected_rdm)


class TestFlattenMatrix(unittest.TestCase):
    def test_flatten_single_matrix(self):
        representational_matrix = np.array([[1, 0.5, 0.6],
                                            [0.5, 1, 0.7],
                                            [0.6, 0.7, 1]])
        representational_vector = flatten_matrix(representational_matrix)
        expected_vector = np.array([[0.5],
                                    [0.6],
                                    [0.7]])
        assert_allclose(expected_vector, representational_vector)

    def test_flatten_multiple_matrices(self):
        representational_matrices = np.array([[[1, 2],
                                              [0.4, 0.7],
                                              [0.5, 0.8]],

                                             [[0.4, 0.7],
                                              [1, 2],
                                              [0.6, 0.9]],

                                             [[0.5, 0.8],
                                              [0.6, 0.9],
                                              [1, 2]]])
        representational_vectors = flatten_matrix(representational_matrices)
        expected_vectors = np.array([[0.4, 0.7],
                                     [0.5, 0.8],
                                     [0.6, 0.9]])
        assert_allclose(expected_vectors, representational_vectors)
