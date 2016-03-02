import unittest

from code.permutation import load, merge


class TestMerge(unittest.TestCase):

    f, b = load()
    movie = merge(f, b)

    def test_merge_returns_not_None(self):
        self.assertIsNotNone(self.movie)

    def test_merge_59_goodman(self):
        self.assertEquals(59, self.movie.John_Goodman.sum())

    def test_merge_goodman_includes_zeros(self):
        self.assertIn(0, self.movie.John_Goodman.unique())
