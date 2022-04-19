## Release notes
---

### 0.9.3 (April 19, 2022)

##### Bug fixes
- When asking Exhibit to generate a specification from a dataset that didn't contain any numerical columns, the resulting specification was missing probability information for categorical columns below the in-line limit.

##### Enhancements
- Revised the specification of custom constraints (previously called conditional constraints). Now you can specify the subset (filter) of the data, the partition, and one or more columns to be affected by the constraints. In addition to the `make_null`, `make_not_null` and `make_outlier`, there are 4 new constraints available: `make_same`, `make_distinct`, `sort_ascending` and `sort_descending`.
- Added an option to generate uuid columns. If your source dataset includes record-level data with unique identifiers, you can exclude them from processing and generate them separately. The uuid columns work differently to normal categorical columns in that you specify the probabilities of the frequency of each unique value appearing in your synthetic dataset. See `uuid_demo.yml` and `uuid_anon.csv` files for examples.
- Added an option to designate numerical columns like age or dose number as categorical. List such columns after the --discrete_columns flag in CLI.

### 0.9.2 (February 14, 2022)

##### Bug fixes
- Fixed a RNG-related bug that could result in slightly different datasets being generated on Linux and Windows from the same specification.

##### Enhancements
- You can now use Exhibit as an importable library, not just as a CLI program. See `recipes/exhibit_scripting.py` for examples of the basic API.
- Exhibit now correctly handles columns composed entirely out of `boolean` values. For the purposes of dataset generation they are treated as categorical rather than numerical values.

### 0.9.1 (December 5, 2021)
Hotfix a Windows-specific bug related to SQLite3 type adaptors.

### 0.9.0 (December 4, 2021)
First beta release ready for limited use in production.