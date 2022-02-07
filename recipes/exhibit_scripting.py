'''
The intended use of Exhibit is via a command line. However, on occasion, you might want to 
incorporate the generation of anonymised datasets into existing Python scripts. This script
demonstrates the basic API that covers both the generation of the specification and the
generation of data.

Run the script from the exhibit/recipes folder
'''

import pandas as pd
from exhibit import exhibit as xbt

datasource_path = "../exhibit/sample/_data/inpatients.csv"
spec_path = "../exhibit/sample/_spec/inpatients_demo.yml"

# SPECIFICATION
original_df = pd.read_csv(datasource_path)

# output="specification" is a scripting-only option that will output a dictionary as it would be
# written to a .YML file so you can make changes before saving it
exhibit_spec = xbt.newExhibit(command="fromdata", source=original_df, output="specification")
spec = exhibit_spec.generate()

print(spec["metadata"]["number_of_rows"]) # => 10225

# DATA

# output="dataframe" is a scripting-only option that will output a Pandas DataFrame as it would be
# exported to a .csv so you can make changes before saving it
exhibit_data = xbt.newExhibit(command="fromspec", source=spec_path, output="dataframe")
anon_df = exhibit_data.generate()

print(anon_df.shape) # => (1448, 9)
