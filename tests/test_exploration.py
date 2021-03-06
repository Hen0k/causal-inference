import sys
sys.path.append('../')
import unittest
import pandas as pd
from scripts.utils import DataLoader
from scripts.exploration import Analysis




class TestAnalysis(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.test_df = DataLoader("tests", "test_data.csv").read_csv()
        self.cleaner = Analysis()


if __name__ == '__main__':
    unittest.main()
