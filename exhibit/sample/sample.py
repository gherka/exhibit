'''
Module referencing sample data for export

inpatients.csv is sourced from ISD Scotland Open Data page:
http://www.isdscotland.org/Health-Topics/Hospital-Care/Inpatient-and-Day-Case-Activity/

prescribing.csv is sourced from NHS Scotland Open Data page:
https://www.opendata.nhs.scot/dataset/prescriptions-in-the-community
'''

# External imports
import pandas as pd
import yaml

# Exhibit imports
from exhibit.core.utils import package_dir

#Load data
inpatients_data = pd.read_csv(package_dir('sample', '_data', 'inpatients.csv'))
prescribing_data = pd.read_csv(package_dir('sample', '_data', 'prescribing.csv'))

#Load specs
with open(package_dir("sample", "_spec", "inpatients_edited.yml")) as f:
    inpatients_spec = yaml.safe_load(f)

with open(package_dir("sample", "_spec", "prescribing.yml")) as f:
    prescribing_spec = yaml.safe_load(f)
