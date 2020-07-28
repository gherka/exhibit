[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Build Status](https://travis-ci.com/gherka/exhibit.svg?branch=master)](https://travis-ci.com/gherka/exhibit) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/gherka/exhibit.svg)](https://lgtm.com/projects/g/gherka/exhibit/context:python) [![Coverage Status](https://coveralls.io/repos/github/gherka/exhibit/badge.svg?branch=master)](https://coveralls.io/github/gherka/exhibit?branch=master)

---
## Exhibit: Command line tool for generating anonymised demonstrator data
---


The purpose of this tool is to make it easier to generate anonymised data in a controlled and reproducible way.

You can specify how many rows of the dummy data to generate, which columns to anonymise and their anonymising patterns, set distribution weights of continuous variables, define derived columns and even determine whether column values have a hierarchical relationship between them. The tool also supports generation of missing data.

The tool has two principal modes of operation: 
 - `fromdata` which produces a detailed, user-editable specification `.yml` file which can be opened by any text editor
 - `fromspec` which produces the anonymised dataset from the supplied specification

See the `-h` listing for the full list of optional command line parameters.

---
### To install:

Download or clone the repository and run `pip install .` from the root folder.

Note that is tool is still in beta so you might want to install it in development mode `-e` and pull the latest version as the tool is updated on GitHub.

---
### Sample data

The repository includes a few sample datasets and exhibit specifications. They are saved in the `exhibit/sample/_data` and `exhibit/sample/_spec` folders respectively. You're welcome to try them out or suggest any other datasets that would be useful to include in the tool.

---
### Database

Exhibit is bundled with a SQLite3 database and a Python utility tool to interact with with it. Alternatively, you can connect directly to `/exhbit/db/anon.db`. The database contains three sample anonymising datasets: `mountains`, `birds` and `patients`.

 - `mountains` has 15 mountain ranges and their top 10 peaks making it useful for anonymising hierarchical pairs, like NHS Boards and Hospitals.
 - `birds` has 150 pairs of common / scientific bird names. This can be useful for 1:1 paired columns.
 - `patients` has 360 made-up patient records with details such as gender, 5-year age band, date of birth and CHI number. Fields from this dataset can be selectively pulled in when linked data is required.

The database is also used to store temporary data for columns where the number of unique values exceeds user threshold and thus not available for editing directly in the `yml` file. Note that this means original, confidential data  might be saved in the `exhibit/db/anon.db` file. You can purge all temporary tables by calling `--purge` command from the included utility tool or by interfacing with the database directly.

---
### Disclaimer

Please note that the degree of anonymisation for each dataset produced by the tool will depend heavily on user choices in the specification. As such, there is no guarantee that confidential data will be suitably masked under all scenarios. If you intend to work with sensitive data, make sure to thoroughly evaluate the output before making it public.
