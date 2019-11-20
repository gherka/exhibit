[![Build Status](https://travis-ci.com/gherka/exhibit.svg?branch=master)](https://travis-ci.com/gherka/exhibit)

---
## Exhibit: Command line tool for generating anonymised demonstrator data
---


The purpose of this tool is to make it easier to generate anonymised data in a controlled manner.

You can specify how many rows of the dummy data to generate, which columns to anonymise and their anonymising patterns, set distribution weights of continuous variables, define derived columns and even determine whether column values have a hierarchical relationship between them. You can also set a random seed to ensure reproducibility.

The tool has two principal modes of operation: 
 - `fromdata` which produces a detailed, user-editable specification `.yml` file which can be opened by any text editor
 - `fromspec` which produces the anonymised dataset

See the `-h` listing for the full list of command line parameters.

---
### To install:

Download or clone the repository and run `pip install .` from the root folder.

Note that is tool is still under development so you might want to install it in development mode `-e` and pull the latest version as the tool is updated on GitHub.

---
### Sample data

The repository includes a few sample datasets and exhibit specifications. They are saved in the `exhibit/sample/_data` and `exhibit/sample_spec` folders respectively. You're welcome to try them out or suggest any other datasets that would be useful to include in the tool.
