import unittest

from code.permutation import load


class TestLoadData(unittest.TestCase):

    features, box_office = load()

    def test_load_returns_not_None(self):
        self.assertIsNotNone(load())

    def test_load_features_shape(self):
        self.assertEquals((39360, 2), self.features.shape)

    def test_load_box_office_shape(self):
        self.assertEquals((8304, 2), self.box_office.shape)
