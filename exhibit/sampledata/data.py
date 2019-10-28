'''
Module referencing sample data for export
'''

# External imports
import pandas as pd

# Exhibit imports
from exhibit.core.utils import package_dir

basic = pd.read_csv(package_dir('sampledata', '_data', 'inpatients.csv'))
