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
    # please review each section and make necessary adjustments.
    # ----------------------------------------------------------
    """)

    yaml_meta = (
        yaml
        .safe_dump(yaml_list[0], sort_keys=False, width=1000)
    )

    c2a = textwrap.dedent("""\
    # ----------------------------------------------------------
    # COLUMN DETAILS
    # ==============
    #
    # Dataset columns are categorised into one of the three types: 
    # Categorical | Continuous | Date
    #
    # Column type determines what parameters are included in the
    # specification. When making changes to the values, please
    # note their format. Values starting with a number must be
    # enclosed in quotes as per YAML rules.
    # 
    # CATEGORICAL COLUMNS
    # -----------
    # The default anonymising method for categorical columns is
    # "random", meaning original values are drawn at random,
    # (respecting probabilities, if supplied) but you can add
    # your own custom sets, including linked, by creating a
    # suitable table in the exhibit database (local or external).
    #
    # The tool comes with a number of sample anonymising sets
    # (see documentation). To use just one column from a set,
    # add a dot separator like so mountains.range
    # 
    # You can also use a subset of regular expression patterns 
    # to generate aliasing values in non-linked columns. For example,
    # if your confidential data had Consultant GMC numbers, you can
    # anonymise them using a regex pattern GMC[0-9]{5}[XY] which
    # will generate values like GMC00000X.
    #
    # Depending on the number of unique values in a column,
    # its original values can either be listed in the spec,
    # put into the exhibit database or, if the values follow a
    # one to one relationship with another column, be listed
    # as part of that column's section.
    #
    # UUID COLUMNS
    # -----------
    # UUID columns are a special case for categorical column where
    # each value is unique, but can appear multiple times depending
    # on the frequency distribution set by the user. You can add UUID
    # columns manually or infer them from the source data. If adding a
    # UUID column manually, don't forget to add it in the metadata section.
    #
    # The format of a UUID column in the specification is as follows:
    # 
    # record_chi:
    #  type: uuid
    #  frequency_distribution:
    #  - frequency | probability_vector
    #  - 1         | 0.5
    #  - 2         | 0.3
    #  - 3         | 0.2
    #  miss_probability: 0.0
    #  anonymising_set: uuid
    #
    # You can choose between uuid and range anonymising sets.
    # ----------------------------------------------------------
    """)

    yaml_columns_all = (
        c2a + yaml.safe_dump(yaml_list[1], sort_keys=False, width=1000))

    c2b = textwrap.dedent("""\
    # ----------------------------------------------------------
    # NUMERICAL COLUMNS
    # ----------
    # Currently, only continuous data is supported. To use numerical
    # columns with discrete data, covert them to categorical or make
    # sure that the precision parameter is set to integer.
    # 
    # To generate Continuous columns you have two distribution options:
    #
    # - weighted_uniform
    # - normal
    #
    # The first option will generate the column by applying weights
    # to a fixed value before scaling it and optionally perturb it
    # within the dispersion percentage. With dispersion set to zero,
    # you can generate identical values for any given combination of
    # categorical values on the row.
    # 
    # A normal distribution will respect categorical weights by
    # shifting the mean accordingly. If dispersion is greater than
    # zero, the value before scaling is perturbed within the 
    # dispersion percentage.
    # 
    # You can also control how values are scaled by setting the
    # distribution target parameters. You must include at least one of
    # target_min, target_max, target_sum, target_mean or target_std.
    #  
    # Note that if you include certain constraints, like ensuring values
    # in one column are always greater than values in another column,
    # the final scaling of the column being adjusted will be affected.
    # ----------------------------------------------------------
    """)

    first_num_col = next(iter(spec_dict["metadata"]["numerical_columns"]), None)

    if first_num_col:
        num_col_in = yaml_columns_all.find(f"  {first_num_col}:\n    type: continuous")
        yaml_columns_all = (
            yaml_columns_all[:num_col_in] + c2b + 
            yaml_columns_all[num_col_in:]
        )

    c2c = textwrap.dedent("""\
    # ----------------------------------------------------------
    # DATE COLUMNS
    # ----------
    # Exhibit will try to determine date columns automatically, but
    # you can also add them manually, providing the following parameters:
    #   type: date
    #   cross_join_all_unique_values: true | false
    #   miss_probability: between 0 and 1
    #   from: 'yyyy-mm-dd'
    #   uniques: 4
    #   frequency: QS
    # 
    # Frequency is based on the frequency strings of DateOffsets.
    # See Pandas documentation for more details. Times are not supported.
    # ----------------------------------------------------------
    """)

    c_geo = textwrap.dedent("""\
    # GEOSPATIAL COLUMNS
    # ----------
    #
    # Geospatial columns are special in that they are not inferred
    # from source data and can only be added manually. The format 
    # of each column is as follows:
    #
    # clinic_coords:
    #    type: geospatial
    #    h3_table: geo_scotland_dz_h3_8
    #    distribution: uniform
    #    miss_probability: 0
    #
    # h3_table refers to H3 hexagon IDs stored in the exhibit DB. exhibit
    # doesn't come with any geospatial lookups by default, but there
    # is a recipe explaining how to create one. The format of the table
    # is that it has to have a column named "h3" with h3 ids. 
    # 
    # For distribution, you can either pick uniform to sample points
    # from all hexagons at random or use column weights in the h3_table,
    # like population counts.
    # ----------------------------------------------------------
    """
    )

    first_date_col = next(iter(spec_dict["metadata"]["date_columns"]), None)

    if first_date_col:
        date_col_in = yaml_columns_all.find(f"  {first_date_col}:\n    type: date")
        yaml_columns_all = (
            yaml_columns_all[:date_col_in] + c2c + c_geo +
            yaml_columns_all[date_col_in:]
        )

    c3 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # CONSTRAINTS
    # ===========
    #
    # There are two types of constraints you can impose of the data:
    # - basic
    # - custom
    # 
    # Basic constraints take the form of a simple statement of the
    # form [dependent_column] [operator] [expression / independent_column].
    # The tool will try to guess these relationships when creating a
    # spec. You can also force a column to be always smaller / larger
    # than a scalar value. Note that adding a basic constraint between
    # two columns will affect the distribution of weights and also the
    # target sum as these are designed to work with a single column.
    #
    # Custom constraints are more flexible and can target specific
    # subsets of values with different actions. Currently the following
    # actions are supported:
    #
    # - "make_null"
    # - "make_not_null"
    # - "make_outlier"
    # - "sort_ascending"
    # - "sort_descending"
    # - "make_distinct"
    # - "make_same"
    # - "make_almost_same"
    # - "generate_as_sequence"
    # - "generate_as_repeating_sequence"
    # - "geo_make_regions"
    # - "sort_and_skew_left"
    # - "sort_and_skew_right"
    # - "sort_and_make_peak"
    # - "sort_and_make_valley"
    # - "shift_distribution_right"
    # - "shift_distribution_left"
    #
    # Adding or banning nulls is useful when a value in one column, 
    # like Readmissions Within 28 days, necessitates a valid value in
    # another, like Readmissions Within 7 days.
    # 
    # The format for custom constraints is as follows:
    #
    # demo_constraint_name:
    #   filter: (los > 2)
    #   partition: age, sex
    #   targets:
    #     taget_column: target_action
    #
    # Custom constraints can target multiple columns, but each column can
    # appear only once in each custom constraint.
    # 
    # Expressions used in the filter must be understood by Pandas eval().
    # Additionally, you can use these custom filters:
    #
    # - "COLUMN_NAME with_high_frequency"
    # - "COLUMN_NAME with_low_frequency"
    #
    # If a column name has spaces, make sure to surround it with
    # the tilde character ~. This rule only applies when columns are used
    # in basic constraints or filters. Targets for custom constraints must 
    # use column names as they are. When comparing a date column against a
    # fixed date, make sure it's in an ISO format and is enclosed in
    # single quotation marks like so '2018-12-01'.
    # ----------------------------------------------------------
    """)

    yaml_constraints = (
        yaml
        .safe_dump(yaml_list[2], sort_keys=False, width=1000)
    )

    c4 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # LINKED COLUMNS
    # ===============
    #
    # There are two types of linked groups - those manually defined using
    # the --linked_columns (or -lc) command line parameter and automatically
    # detected groups where columns follow a hierarchical (one to many)
    # relationship. User defined linked columns are always put under the
    # zero indexed group.
    #
    # Groups of hierarchically linked columns are listed together under the 
    # index starting from 1. Their details are saved in the exhibit database.
    #
    # The specification format is as follows:
    # - - 1
    #   - - Parent column
    #   - - Child column
    # - - 2
    #   - - ...etc.
    # It's possible to add a linked columns group manually by adding 
    # a table to the exhibit databse with the hierarchical relationships.
    # The name of this table must follow the format: id_N  where id is 
    # taken from the metadata section and N is the group number.
    # ----------------------------------------------------------
    """)

    yaml_linked = yaml.safe_dump(yaml_list[3], sort_keys=False, width=1000)

    c5 = textwrap.dedent("""\
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
    # For example, assuming you have Numerator column A and
    # Denominator column B, you would write Rate: (A / B)
    # ----------------------------------------------------------
    """)

    yaml_derived = yaml.safe_dump(yaml_list[4], sort_keys=False, width=1000)

    c6 = textwrap.dedent("""\
    # ----------------------------------------------------------
    # MODELS
    # ===============
    # 
    # You can generate columns in your dataset using custom machine
    # learning models, using the synthetic dataset generated so far
    # as input. Follow the tutorial in the recipes folder to create
    # your model and save it in the models folder. The specification
    # format is as follows:
    # 
    # models:
    #   model_name: (without .pickle extension)
    #     hyperparameter_name : hyperparameter_value
    # 
    # You can chain models one after another - they are called in the
    # same order as they appear in the specification. Make sure that
    # the same libraries used in creating the model are available in
    # the environment where Exhibit is installed.
    # ----------------------------------------------------------
    """)

    yaml_models = yaml.safe_dump(yaml_list[5], sort_keys=False, width=1000)

    spec_yaml = (
        c1 + yaml_meta + yaml_columns_all + c3 + yaml_constraints +
        c4 + yaml_linked + c5 + yaml_derived + c6 + yaml_models)

    # replace empty lists / dicts with a blank
    spec_yaml = spec_yaml.replace(r" []", "").replace(r" {}", "")

    return spec_yaml
