'''
The intended use of Exhibit is via a command line. However, on occasion, you might want to 
incorporate the generation of anonymised datasets into existing Python scripts. This script
demonstrates the basic API that covers both the generation of the specification and the
generation of data.

Run the script from the exhibit/recipes folder
'''

import pandas as pd
from exhibit import exhibit as xbt
from exhibit.core.spec import (
    Spec, UUIDColumn, CategoricalColumn, NumericalColumn, DateColumn)

datasource_path = "../exhibit/sample/_data/inpatients.csv"
spec_path = "../exhibit/sample/_spec/inpatients_demo.yml"

# SPECIFICATION
# 1. FROM CSV
original_df = pd.read_csv(datasource_path)

# output="specification" is a scripting-only option that will output a dictionary as it would be
# written to a .YML file so you can make changes before saving it
exhibit_spec = xbt.Exhibit(command="fromdata", source=original_df, output="specification")
spec = exhibit_spec.generate()

print(spec["metadata"]["number_of_rows"]) # => 10225

# 2. FROM SCRATCH
spec = Spec()
spec_dict = spec.generate()

spec_dict["metadata"]["number_of_rows"] = 1000
spec_dict["metadata"]["uuid_columns"] = ["id"]
spec_dict["metadata"]["categorical_columns"] = ["hospital"]
spec_dict["metadata"]["numerical_columns"] = ["count"]
spec_dict["metadata"]["date_columns"] = ["discharge_date"]

spec_dict["columns"]["id"] = UUIDColumn(uuid_seed=0)
spec_dict["columns"]["hospital"] = CategoricalColumn("hospital", original_values="regex", anon_set="HOSP[1-9]{2}")
spec_dict["columns"]["count"] = NumericalColumn(distribution_parameters={"target_min":1, "target_max":1000})
spec_dict["columns"]["discharge_date"] = DateColumn("2020-01-01", 360 * 2, cross_join=False)

exhibit_data = xbt.Exhibit(command="fromspec", source=spec_dict, output="dataframe")
anon_df = exhibit_data.generate()

print(anon_df.shape) # => (1000, 4)

# DATA

# output="dataframe" is a scripting-only option that will output a Pandas DataFrame as it would be
# exported to a .csv so you can make changes before saving it
exhibit_data = xbt.Exhibit(command="fromspec", source=spec_path, output="dataframe")
anon_df = exhibit_data.generate()

print(anon_df.shape) # => (1448, 9)
