## Exhibit: Command line tool to create anonymised demonstrator data

The goal of Exhibit is to make it easier to generate synthethic data at scale in a controlled and reproducible way.

---

<table>
<!-- Main branch badges -->
<tr>
<td style="text-align: left"><code>main</code></td>
<td style="text-align: left; vertical-align: middle; padding: 4px; line-height: 1">
<p style="margin:0">
<a href="https://travis-ci.org/gherka/exhibit"><img src="https://travis-ci.org/gherka/exhibit.svg?branch=main" alt="Build Status"> <a href="https://github.com/gherka/exhibit/actions/workflows/codeql.yml"><img src="https://github.com/gherka/exhibit/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"> <a href="https://codecov.io/github/gherka/exhibit"><img src="https://codecov.io/github/gherka/exhibit/branch/main/graph/badge.svg?token=A5YQIV56QM" alt="codecov">
</p>
</td>
</tr>
<!-- Latest branch badges -->
<tr>
<td style="text-align: left;"><code>latest</code></td>
<td style="text-align: left; vertical-align: middle; padding: 4px; line-height: 1">
<p style="margin:0">
<a href="https://travis-ci.org/gherka/exhibit"><img src="https://travis-ci.org/gherka/exhibit.svg?branch=latest" alt="Build Status"> <a href="https://github.com/gherka/exhibit/actions/workflows/codeql.yml"><img src="https://github.com/gherka/exhibit/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"> <a href="https://codecov.io/github/gherka/exhibit"><img src="https://codecov.io/github/gherka/exhibit/branch/latest/graph/badge.svg?token=A5YQIV56QM" alt="codecov">
</p>
</td>
</tr>

</table>

## Key features

- Control all aspects of the anonymisation process: which columns to anonymise and to what degree
- Rapidly iterate on the anonymisation options
- Set categorical weights to create custom distributions
- Use regular expressions to bulk-anonymise identifiers
- Add columns derived from newly anonymised data
- Preserve important relationships between your columns (paired, hierarchical, custom)
- Add outliers to any subset of the generated data
- Generate and manipulate missing data and timeseries
- Generate geo-spatial data using H3 hexes
- Augment your synthetic data with compiled machine learning models and custom functions
- Use SQL to generate conditional values based on external tables

## Installation:

To install using pip, enter the following command at a Bash or Windows command prompt:

`pip install exhibit`

Alternatively, download or clone the repository and run `pip install .` from the root folder.

## Quickstart

Exhibit has two principal modes of operation: 
 - `fromdata` produces a detailed, user-editable `.yml` specification
 - `fromspec` which produces the anonymised dataset from the supplied specification

See the `-h` listing for the full list of optional command line parameters.

The repository includes a few sample datasets and specifications.\
You can find them in `exhibit/sample/_data` and `exhibit/sample/_spec`

To create a demo dataset, run:\
`exhibit fromspec exhibit/sample/_spec/inpatients_demo.yml -o demo.csv`

To create a demo specification that equialises all probabilities and weights, run:\
`exhibit fromdata exhibit/sample/_data/inpatients.csv -ew -o demo.yml`

## Database

Exhibit is bundled with a SQLite3 database and a Python utility tool to interact with it. Alternatively, you can connect directly to `/exhbit/db/exhibit.db`. The database contains three sample aliasing datasets: `mountains`, `birds` and `patients` designed to help you quickly alias original values without manually editing individual column values.

 - `mountains` has 15 mountain ranges and their top 10 peaks making it useful for aliasing hierarchical pairs, like NHS Boards and Hospitals.
 - `birds` has 150 pairs of common / scientific bird names. This can be useful for 1:1 paired columns.
 - `patients` has 360 made-up patient records with details such as gender, 5-year age band, date of birth and CHI number. Fields from this dataset can be selectively pulled in when linked data is required.
  - `dates` has dates ranging from 1900-01-01 to 2100-01-01 at a single day interval. This table is useful if you have a SQL statement in the `anonymising_set` that picks dates based on a condition. 

The database is also used to store temporary data for columns where the number of unique values exceeds user threshold and thus not available for editing directly in the `yml` file.

**Note that original, confidential data might be saved in the `exhibit/db/exhibit.db` file on your local machine. You can purge all temporary tables by calling `--purge` command from the included utility tool or by interfacing with the database directly.**

## Disclaimer

Please note that the degree of anonymisation for each dataset produced by the tool will depend heavily on user choices in the specification. As such, there is no guarantee that confidential data will be suitably masked under all scenarios. If you intend to work with sensitive data, make sure to thoroughly evaluate the output before making it public.
