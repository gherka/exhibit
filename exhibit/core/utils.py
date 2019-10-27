'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
import re
from itertools import chain, combinations
import datetime
import dateutil
import textwrap

# External library imports
import pandas as pd
import yaml

def build_list_of_original_values(series):
    '''
    Return a padded list of strings
    '''
    original_values = sorted(series.unique().tolist())
    longest = len(max(original_values, key=len))

    padded_values = [x.ljust(longest + 1) for x in original_values]

    return padded_values


def build_list_of_probability_vectors(series, total_count):
    '''
    Return a list of probability vectors as strings
    '''
    vectors = (series
        .value_counts()
        .sort_index(kind="mergesort")
        .apply(lambda x: x / total_count)
        .values
        .tolist())

    string_vectors = ["{0:.3f} ".format(x) for x in vectors]

    return string_vectors

def build_list_of_column_weights(weights):
    '''
    weights is a dictionary {col_name: list_of_weights}
    '''

    sorted_weights = [weights[key] for key in sorted(weights)]

    sorted_final = [" | ".join(
        ["{0:.3f}".format(y) for y in x])
        for x in zip(*sorted_weights)]

    return sorted_final
    

def build_table_from_lists(series, total_count, weights):
    '''
    Doc string
    '''
    s1 = build_list_of_original_values(series)
    s2 = build_list_of_probability_vectors(series, total_count)
    s3 = build_list_of_column_weights(weights)

    final = ["| ".join(x) for x in zip(s1, s2, s3)]

    return final

def path_checker(string):
    '''
    Improves error message for user if wrong path entered.
    Returns Path object.
    '''
    if not exists(string):
        msg = "Can't find specified file"
        raise FileNotFoundError(msg)
    return Path(string)
    
def package_dir(*args):
    '''
    Returns absolute path to package / package modules / files
    given names relative to the package root directory

    __file__ attribute  is the pathname of the file from
    which the module was loaded; each module using this
    function will take its own file path from the global
    namespace. Dot dot just moves it up one level which
    imposes certain constrains of the file structure of
    the project.
    '''
    return abspath(join(dirname(__file__), "..", *args))

def date_parser(row_tuple):
    '''
    Assuming row_tuple has the form (column_name, row_value)
    check if row_value has date separators and then
    as a back-up run it through the dateutil parser
    (as it can throw up a lot of false positives).
    '''
    column_name, row_value = map(str, row_tuple)
    if re.search(r'[-:/]', row_value):
        try:
            dateutil.parser.parse(row_value)
            return column_name
        except ValueError:
            pass

def read_with_date_parser(path):
    '''
    Adapt the read_csv function of Pandas to
    detect and parse datetime columns.
    '''

    if path.suffix in ['.csv',]:

        df = pd.read_csv(path)

        for x in df.loc[0, :].iteritems():
            time_col = date_parser(x)
            if not time_col is None:
                df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
                
        return df
    
    raise TypeError("Only .csv file format is supported")

def generate_YAML_string(spec_dict):
    '''
    Returns a string formatted to a YAML spec
    from a passed in dictionary

    We overwrite ignore_aliases() to output identical dictionaries
    and not have them replaced by aliases like *id001
    '''

    yaml.SafeDumper.ignore_aliases = lambda *args: True

    yaml_list = [{key:value} for key, value in spec_dict.items()]

    c1 = textwrap.dedent("""\
    #---------------------------------------------------------
    #This specification describes the dataset in great detail.
    #In order to vary to degree to which it is anonymised,
    #please review the sections and make necessary adjustments
    #---------------------------------------------------------
    """)

    yaml_meta = yaml.safe_dump(yaml_list[0], sort_keys=False)

    c2 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Dataset columns can be one of the three types: 
    #Categorical | Continuous | Timeseries
    #Column type determines the parameters in the specification
    #When making changes to the values, please note their format.
    #---------------------------------------------------------
    """)

    yaml_columns = yaml.safe_dump(yaml_list[1], sort_keys=False)

    c3 = textwrap.dedent("""\
    #---------------------------------------------------------
    #The tool will try to guess which columns are "linked",
    #meaning that values cascade from one column to another.
    #If any grouping is missed, please add it manually.
    #---------------------------------------------------------
    """)

    yaml_constraints = yaml.safe_dump(yaml_list[2], sort_keys=False)

    c4 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Please add any rates to be calculated from anonymised
    #numerator and denominator in this section, alongside with
    #the calculation used. The tool will automatically include
    #columns with the word "Rate" in the column name here and
    #the defaul calculation is a random float between 0 and 1.
    #---------------------------------------------------------
    """)

    yaml_derived = yaml.safe_dump(yaml_list[3], sort_keys=False)

    c5 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Please add any demonstrator patterns in this section.
    #---------------------------------------------------------
    """)

    yaml_demo = yaml.safe_dump(yaml_list[4], sort_keys=False)
    
    spec_yaml = (
        c1 + yaml_meta + c2 + yaml_columns + c3 + yaml_constraints +
        c4 + yaml_derived + c5 + yaml_demo)

    return spec_yaml

def guess_date_frequency(timeseries):
    '''
    Try to guess if the sorted timestamps have any pattern to them.
    
    Pandas diff() on the sorted duplicate-less datafraeme computes
    the difference between each element with its previous row which
    gives as the time lapsed between discrete time stamps. 

    We then look at how many such differences exist and what their values
    are in days.

    If the period between two unique timestamps is between 28 and 31 days
    then we guess it's a monthly timerseries and so on.

    See description of time alises on Pandas website.
    '''
    
    time_diff_counts = (timeseries
                        .drop_duplicates()
                        .sort_values()
                        .diff()
                        .value_counts())
    
    if len(time_diff_counts.index) == 1:

        if time_diff_counts.index[0].days == 1:
            return "D"        
        elif time_diff_counts.index[0].days in range(28, 32):
            return "MS"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "QS"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "YS"
    
    elif time_diff_counts.index[0].days - time_diff_counts.index[1].days in range(0, 3):
        
        if time_diff_counts.index[0].days == 1:
            return "D"
        elif time_diff_counts.index[0].days in range(28, 32):
            return "MS"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "QS"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "YS"
        
    else:
        return None

def get_attr_values(spec_dict, attr, col_names=False, types=None):
    '''
    spec_dict should be YAML de-serialised into
    dictionary.

    Assuming the spec was generated correctly,
    go through all columns and capture given
    attribute's value; None if attribute is 
    missing.
    
    Returns a list with values
    from columns in order of appearance in the
    spec.

    Optional argument to return a col_name, attribute value
    instead of just a list of attribute values
    '''
    
    if types is None:
        types = ['categorical', 'date', 'continuous']
    
    if not isinstance(types, list):
        types = [types]

    attrs = []

    if col_names:

        for col in spec_dict['columns']:
        #append None as a placeholder; overwrite if attr exists
            if spec_dict["columns"][col]['type'] in types:
                attrs.append((col, None))
                for a in spec_dict['columns'][col]:
                    if a == attr:
                        attrs[-1] = (col, spec_dict['columns'][col][attr])

    else:
        for col in spec_dict['columns']:
            if spec_dict["columns"][col]['type'] in types:
                attrs.append(None)
                for a in spec_dict['columns'][col]:
                    if a == attr:
                        attrs[-1] = spec_dict['columns'][col][attr]
    return attrs

def find_linked_columns(df):
    '''
    Given a dataframe df, return a list
    of tuples with column names where values in 
    one column are always paired with the 
    same value in another, as in, for example,
    an NHS Board and NHS Board Code.

    Columns with the same value in all rows are skipped.

    The column that maps 1 to many is appended first,
    juding by the number of unique values it has.
    '''
    linked = []
    
    cols = [col for col in df.columns if df[col].nunique() > 1]
    
    for col1, col2 in combinations(cols, 2):

        if ( 
                df.groupby(col1)[col2].nunique().max() == 1 or
                df.groupby(col2)[col1].nunique().max() == 1
            ): 
            
            if df[col1].nunique() <= df[col2].nunique():

                linked.append((col1, col2))

            else:

                linked.append((col2, col1))

    return linked

class linkedColumnsTree:
    '''
    Organizes a list of tuples into a matrix of sorts
    where each row is a list of columns ordered from
    ancestor to descendants.
    
    connection tuples should come in ancestor first.

    Think of a good test to make sure this class
    is working as intended.
    
    '''

    def __init__(self, connections):
        '''
        Main output of the class is stored in the tree attribute
        as a list of tuples of the form (linked columns group number,
        list of grouped columns)
        '''
        self._tree = []
        self.add_connections(connections)
        self.tree = [(i, list(chain(*x))) for i, x in enumerate(self._tree)]

    def add_connections(self, connections):
        '''
        Doc string
        '''
        for node1, node2 in connections:
            self.add_node(node1, node2)
        
    def find_element_pos(self, e, source_list):
        '''
        The tree is a list implementation of a (mutable)
        matrix and finding position involves simply
        traversing i and j dimensions.
        '''
        #vertical position is i
        for i, l in enumerate(source_list):
            #horizontal position is j
            for j, sublist in enumerate(l):
                if e in sublist:
                    return (i, j)
        return None
    
    def add_node(self, node1, node2):
        '''
        There are three possible outcome when processing a
        linked columns tuple:

        - Neither columns are in the tree already
        - Ancestor (node1) column is in the tree
        - Descendant (node2) column is in the tree

        The fourth possible situation, both columns already
        in the tree, isn't possible because duplicates are
        filtered out by find_linked_columns.
        
        '''
        
        apos = self.find_element_pos(node1, self._tree)
        dpos = self.find_element_pos(node2, self._tree)
        
        #add a new completely new row with both nodes
        if (apos is None) & (dpos is None):
            self._tree.append([[node1], [node2]])
        
        #if found ancestor in the tree, add descendant
        elif (not apos is None) & (dpos is None):
            ai, aj = apos
            if isinstance(self._tree[ai][aj], list):
                self._tree[ai][aj+1].append(node2)
            else:
                self._tree[ai][aj+1].append([node2])
                
        #if found descendant in the tree, add ancestor
        elif (apos is None) & (not dpos is None):
            di, dj = dpos
            if isinstance(self._tree[di][dj], list):
                self._tree[di][dj-1].append(node1)
            else:
                self._tree[di][dj-1].append([node1])

def generate_id():
    '''
    Generate a 5-digit pseudo-unique ID based on current time
    '''
    new_id = str(hex(int(datetime.datetime.now().timestamp()*10))[6:])

    return new_id
