import unittest

import pandas as pd

from code.permutation import _to_arr, _tstat


df = pd.DataFrame({'movie_id' : [1, 23, 45, 67, 89],
                   'John_Goodman' : [0, 1, 0, 1, 0],
                   'hit' : [1, 1, 0, 0, 1]})
arr, n = _to_arr(df, 'John_Goodman')

class TestToArray(unittest.TestCase):

    def test_to_arr_order(self):
        self.assertTrue((arr == [1, 0, 1, 0, 1]).all())

    def test_arr_n(self):
        self.assertEquals(2, n)

class TestTstat(unittest.TestCase):

    def test_tstat_17(self):
        self.assertEquals(0.17, round(_tstat(arr, n), 2))
