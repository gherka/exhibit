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
inpatients_data = pd.read_csv(package_dir("sample", "_data", "inpatients.csv"))
inpatients_anon = pd.read_csv(
    package_dir("sample", "_data", "inpatients_anon.csv"),
    parse_dates=["quarter_date"])

uuid_anon = pd.read_csv(
    package_dir("sample", "_data", "uuid_anon.csv"),
    parse_dates=["vacc_date"])

prescribing_data = pd.read_csv(
    package_dir("sample", "_data", "prescribing.csv"), parse_dates=["PaidDateMonth"])

#Load specs
with open(package_dir("sample", "_spec", "inpatients_demo.yml")) as f:
    inpatients_spec = yaml.safe_load(f)
