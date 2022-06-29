import sys
sys.path.append('../')
import unittest
import pandas as pd
from scripts.utils import DataLoader
from scripts.cleaning import CleanDataFrame




class TestCleanDataFrame(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.test_df = DataLoader("tests", "test_data.csv").read_csv()
        self.cleaner = CleanDataFrame()

    def test_remove_null_row(self):
        df = self.cleaner.remove_null_row(self.test_df, self.test_df.columns)
        self.assertEqual(len(df), 3)


if __name__ == '__main__':
    unittest.main()
