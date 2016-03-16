import unittest

import numpy as np
import scipy as sp
import pandas as pd

from code.sedc import *


class TestSEDC(unittest.TestCase):

    coefficients = load_coefficients()
    reviews = load_reviews()
    X, feature_names = features(reviews)
    w = np.array([0.5, -0.5])
    v = np.array([1, 1])

    def test_load_coefficients_returns_not_None(self):
        self.assertIsNotNone(load_coefficients())

    def test_load_coefficients_type_df(self):
        assert isinstance(self.coefficients, pd.DataFrame)

    def test_load_coefficients_shape(self):
        self.assertEquals((10001, 2), self.coefficients.shape)

    def test_load_reviews_returns_not_None(self):
        self.assertIsNotNone(load_reviews())

    def test_load_reviews_type_ndarray(self):
        assert isinstance(self.reviews, np.ndarray)

    def test_load_reviews_len_10(self):
        self.assertEquals(10, len(self.reviews))

    def test_features_returns_not_None(self):
        self.assertIsNotNone(features(self.reviews))

    def test_features_names_type_list(self):
        assert isinstance(self.feature_names, list)

    def test_features_X_type_sparse_csr(self):
        assert isinstance(self.X, sp.sparse.csr.csr_matrix)

    def test_features_number_of_features(self):
        self.assertEquals(len(self.feature_names), self.X.shape[1])

    def test_df_tdm_w_returns_not_None(self):
        self.assertIsNotNone(df_tdm_w(self.coefficients, self.X,
                                      self.feature_names))

    def test_logistic_returns_not_None(self):
        self.assertIsNotNone(logistic(self.w, self.v))

    def test_logistic_is_half(self):
        self.assertEquals(0.5, logistic(self.w, self.v))
