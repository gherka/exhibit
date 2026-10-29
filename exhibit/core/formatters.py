'''
A collection of converters to help with outputting
and reading back user specification
'''
# Standard library imports
from itertools import zip_longest

# External library imports
import pandas as pd
import pandas.api.types as ptypes

# Exhibit imports
from exhibit.core.constants import MISSING_DATA_STR

class FormattedList(list):
    '''
    A special type of list to help identify formatted list of strings
    used to construct a csv-like table in the spec. This way, we can
    separate processing for these formatted values from a basic list
    of values passed to original_values during manual column creation
    '''


def format_header(dataframe, series_name, prefix=None):
    '''
    Function to pad the header values based on the length
    of the header column's values

    Applies only to categorical columns with original_values

    Parameters
    ----------
    dataframe : pd.DataFrame
        source dataframe
    series_name : str
        Name of the series whose header we're formatting
    prefix : str
        Prefix to use to identify paired columns

    Returns
    -------
    Formatted string value of series_name
    '''

    series = dataframe[series_name].map(str).unique()

    if prefix:
        series_name = prefix + series_name

    longest = max(
        len(MISSING_DATA_STR),
        len(series_name),
        len(max(series, key=len))
    )

    return series_name.ljust(longest)

def _floor_for_display(x):
    '''
    Values are shown to 3 decimal places so anything below 0.001 would be displayed as
    0.000 and, once read back, never be picked during generation. To avoid it, non-zero
    values are floored at 0.001. Zero stays zero so that, for example, an unused MISSING
    row remains unused.
    '''

    return 0 if x == 0 else max(0.001, x)

def format_list_of_probability_vectors(probas):
    '''
    Format probabilities for display in the YAML table; see _floor_for_display.

    Parameters
    ----------
    probas : iterable of float
        raw probabilities, as stored in the probability_vector column

    Returns
    -------
    probability_vector formatted into a list of padded strings
    '''

    HEADER = "probability_vector"

    return [
        f"{_floor_for_display(x):.3f}".ljust(len(HEADER)) for x in probas
    ]

def parse_original_values(original_values):
    '''
    Parses the value of the original_values attribute of the spec
    for downstream processing. Can either be a list of strings with each
    element being a row of the .csv-like table or a plain string.

    Parameters
    ----------
    original_values : list or str or DataFrame
        If list, the first element is the header row. str could be regex, paired column
        indicator, etc. DataFrame could be passed from internal testing.

    Because the original_table is constructed with a lot of padding,
    each value in the list has to be stripped of spaces.

    The separator character between .csv-like table values is |

    The only types we're likely to encounter in the original_table
    are strings and floats.

    Importantly, we don't rescale the probabilities at this point and delegate it
    to where we're actually calling rng.choice() where probabilties MUST sum up to 1.

    The reason for it is some custom actions, like generate_as_sequence, use
    probabilities in a different way to normal generation (combined probabilities).

    Returns
    -------
    Pandas DataFrame or untouched string
    '''

    if isinstance(original_values, list):

        df = pd.DataFrame(
            data=[
                map(str.strip, x.split("|")) for x in original_values[1:]
            ],
            columns=[x.strip() for x in original_values[0].split("|")],
        )

        df["probability_vector"] = df["probability_vector"].astype(float)

        return df

    return original_values

def format_uuid_frequency_distribution(freq_df):
    '''
    Formats a frequency distribution (as produced by
    UUIDColumn.calculate_frequency_distribution) into a list of strings suitable for
    YAML rendering: a header row followed by "frequency | probability" rows.
    Padding is automatically adjusted by ljust even if the frequency goes
    into double digits.
    '''
    header = ["frequency | probability_vector"]

    freq_list = (
        freq_df["frequency"].astype(str).str.ljust(9)
        .str.cat(freq_df["probability_vector"].transform(lambda x: f"{x:.3f}"), sep=" | ")
        .tolist()
    )

    return header + freq_list

def format_original_values(df):
    '''
    Render an original_values table (as exported by CategoricalColumn.to_dict) into
    the csv-like list of padded strings used in the YAML spec.

    Expected layout, left to right: the column's own values, any "paired_<name>"
    columns, probability_vector, then one weight column per numerical column. The
    layout is taken from the DataFrame itself so the same function serves plain,
    paired and weightless columns. Numerical weight columns are sorted by name to match
    the legacy output.

    Every cell is padded to the width of its column and each row ends with " |", which is
    what parse_original_values expects to read back.

    Parameters
    ----------
    df : pd.DataFrame
        original_values (see CategoricalColumn._build_original_values_df)

    Returns
    -------
    FormattedList
    '''

    cols        = list(df.columns)
    prob_idx    = cols.index("probability_vector")
    value_cols  = cols[:prob_idx]
    weight_cols = sorted(cols[prob_idx + 1:])

    table = {}

    for col in value_cols:
        values = df[col].astype(str).tolist()
        width  = max([len(col), len(MISSING_DATA_STR)] + [len(x) for x in values])
        table[col] = [col.ljust(width)] + [x.ljust(width) for x in values]

    table["probability_vector"] = (
        ["probability_vector"] +
        format_list_of_probability_vectors(df["probability_vector"].tolist())
    )

    for col in weight_cols:
        # 5 is the width of a formatted weight, e.g. 0.123
        width = max(5, len(col))
        table[col] = (
            [col.ljust(width)] +
            [f"{_floor_for_display(x):.3f}".ljust(width) for x in df[col]]
        )

    ordered = value_cols + ["probability_vector"] + weight_cols

    return FormattedList(" | ".join(row) + " |" for row in zip(*(table[c] for c in ordered)))

def format_df_for_export(df):
    '''
    The way data is formatted is mainly controlled by the DataFrame formatting
    classes. There is no easy way to modify it for our purposes so this is a
    semi-manual way of cleaning up / hiding any inconsistencies. This is a last
    resort method so try to solve any formatting / typing issues elsewhere.
    '''

    for column in df.columns:
        if ptypes.is_timedelta64_dtype(df[column]):
            df[column] = df[column].astype(str).str.replace("0 days ", "")

    return df
