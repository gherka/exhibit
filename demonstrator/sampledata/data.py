'''
Module generating sample data
'''

# External imports
import pandas as pd

# Demonstrator imports
from demonstrator.core.utils import package_dir

basic = pd.read_csv(package_dir('sampledata', '_data', 'basic.csv'))
