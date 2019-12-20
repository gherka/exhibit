[![Build Status](https://travis-ci.com/gherka/exhibit.svg?branch=master)](https://travis-ci.com/gherka/exhibit)

---
## Exhibit: Command line tool for generating anonymised demonstrator data
---


The purpose of this tool is to make it easier to generate anonymised data in a controlled and reproducible way.

You can specify how many rows of the dummy data to generate, which columns to anonymise and their anonymising patterns, set distribution weights of continuous variables, define derived columns and even determine whether column values have a hierarchical relationship between them.

The tool has two principal modes of operation: 
 - `fromdata` which produces a detailed, user-editable specification `.yml` file which can be opened by any text editor
 - `fromspec` which produces the anonymised dataset

See the `-h` listing for the full list of optional command line parameters.

---
### To install:

Download or clone the repository and run `pip install .` from the root folder.

Note that is tool is still in beta so you might want to install it in development mode `-e` and pull the latest version as the tool is updated on GitHub.

---
### Sample data

The repository includes a few sample datasets and exhibit specifications. They are saved in the `exhibit/sample/_data` and `exhibit/sample_spec` folders respectively. You're welcome to try them out or suggest any other datasets that would be useful to include in the tool.

---
### Database

Exhibit is bundled with a SQLite3 database and a Python utility tool to interact with with it. Alternatively, you can connect directly to `/exhbit/db/anon.db`. The database contains two sample anonymising datasets: `mountains` and `birds`. The former has 15 mountain ranges and their top 10 peaks making it useful for anonymising hierarchical pairs, like NHS Boards and Hospitals. `birds` is simply 150 pairs of common / scientific bird names. This can be useful for 1:1 paired columns. The database also creates temporary tables to store columns where the number of unique values exceeds user threshold. 

---
### Disclaimer

Please note that the degree of anonymisation for each dataset produced by the tool will depend heavily on user choices in the specification. As such, there is no guarantee that confidential data will be suitably masked under all scenarios. If you intend to work with sensitive data, make sure to thoroughly evaluate the output before making it public.
