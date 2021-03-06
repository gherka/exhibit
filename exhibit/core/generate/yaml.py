'''
Format YAML spec and organise comments
'''

# Standard library imports
import textwrap

# External library imports
import yaml

# EXPORTABLE METHODS
# ==================
def generate_YAML_string(spec_dict):
    '''
    Serialise specification dictionary into a YAML string with added comments

    Paramters
    ---------
    spec_dict : dict
        complete specification of the source dataframe
    
    Returns
    -------
    YAML-formatted string

    We overwrite ignore_aliases() to output identical dictionaries
    and not have them replaced by aliases like *id001
    '''
    
    yaml.SafeDumper.ignore_aliases = lambda *args: True

    yaml_list = [{key:value} for key, value in spec_dict.items()]

    c1 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # EXHIBIT SPECIFICATION
    # =====================
    #
    # This specification describes the dataset in great detail.
    # In order to vary the degree to which it is anonymised,
    # please review each section and make necessary adjustments
    # ----------------------------------------------------------
    """)

    yaml_meta = yaml.safe_dump(yaml_list[0], sort_keys=False, width=1000)

    c2 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # COLUMN DETAILS
    # ==============
    #
    # Dataset columns can be one of the three types: 
    # Categorical | Continuous | Timeseries
    #
    # Column type determines what parameters are included in the
    # specification. When making changes to the values, please
    # note their format. Values starting with a number must be
    # enclosed in quotes as per YAML rules.
    #
    # The default anonymising method for categorical columns is
    # "random", meaning original values are drawn at random,
    # (respecting probabilities, if supplied) but you can add
    # your own custom sets, including linked, by creating a
    # suitable table in the anon.db SQLite3 database.
    #
    # The tool comes with a number of sample anonymising sets
    # (see documentation). To use just one column from a set,
    # add a dot separator like so mountains.range
    #
    # For Continuous columns you have two options, each with
    # their own set of distinct parameters:
    #
    # - weighted_uniform_with_dispersion
    # - normal
    #
    # The first option will generate the column by appling weights
    # to the uniform_base_value parameter and optionally perturb it
    # within the dispersion percentage. With dispersion set to zero, you
    # can generate identical values for any given combination of
    # categorical values on the row.
    # 
    # A normal distribution will respect categorical weights by shifting
    # the mean accordingly. You can vary the "spread" by adjusting the 
    # std parameter.
    # ----------------------------------------------------------
    """)

    yaml_columns = yaml.safe_dump(yaml_list[1], sort_keys=False, width=1000)

    c3 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # CONSTRAINTS
    # ===========
    #
    # The tool will try to guess which columns are "linked".
    # The meaning of "linked" varies depending on whether the
    # columns are categorical or numerical.
    #
    # For linked categorical columns, values in one column must
    # map 1 : many to values in another column. The columns are
    # listed in descending order from parent to child/children.
    #
    # For linked numerical columns, all non-null values in one
    # column must be smaller / larger than corresponding row values
    # in another column.
    #
    # If a column name has spaces, make sure to surround it with
    # the tilde character ~. You can also force a column to be
    # always smaller / larger than a scalar value. Note that adding
    # a boolean contraint between two columns will affect the 
    # distribution of weights and also the target sum as these are
    # designed to work with a single, discrete column. When comapring
    # a date column against a fixed date, make sure it's in an ISO
    # format and is enclosed in single quites like so: '2018-12-01'.
    # ----------------------------------------------------------
    """)

    yaml_constraints = yaml.safe_dump(yaml_list[2], sort_keys=False, width=1000)

    c4 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # DERIVED COLUMNS
    # ===============
    #
    # You can add columns that will be calculated after the rest
    # of the dataset has been generated.
    #
    # The calculation should be in a format that Pandas' eval()
    # method can parse and understand. 
    #
    # For examle, assuming you have Numerator column A and
    # Denomininator column B, you would write Rate: (A / B)
    # ----------------------------------------------------------
    """)

    yaml_derived = yaml.safe_dump(yaml_list[3], sort_keys=False, width=1000)
    
    spec_yaml = (
        c1 + yaml_meta + c2 + yaml_columns + c3 + yaml_constraints +
        c4 + yaml_derived)

    return spec_yaml
