'''
Module isolating methods and classes to find, process and generate linked columns
'''

# Standard library imports
from itertools import chain, combinations
from typing import Tuple, List
from collections import deque, defaultdict

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from .utils import exceeds_ct
from .sql import query_anon_database

# EXPORTABLE METHODS & CLASSES
# ============================
class LinkedColumnsTree:
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
        as a list of tuples of the form:
        (linked columns group number,
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

def generate_linked_anon_df(spec_dict, linked_group: Tuple[int, List[str]], num_rows):
    '''
    Create a dataframe for a SINGLE linked group

    Parameters
    ----------
        spec_dict : dict
            YAML specification de-serialised into a dictionary
        linked_group : tuple
            tuple consisting of linked group number and a list of linked columns
        num_rows : number
            how many rows to generate
    
    Returns
    -------
    Linked dataframe
    '''  

    gen = _LinkedDataGenerator(spec_dict, linked_group, num_rows)

    linked_df = gen.pick_scenario()

    return linked_df

def find_hierarchically_linked_columns(df, spec):
    '''
    Given a dataframe df, return a list
    of tuples with column names where values in 
    the second column are always paired with the 
    same value in the first column (many:1 relationship)
    '''

    linked = []

    #drop NAs because replacing them with Missing data means
    #that columns that are normally linked, won't be (Missing data will
    #appear for multiple "parent" columns)
    df = df.select_dtypes(exclude=np.number).dropna()
    
    #single value and paired columns are ignored
    cols = []

    for col in spec['metadata']['categorical_columns']:
        cond = (
            (df[col].nunique() > 1) &
            (spec['columns'][col]['original_values'] != "See paired column")
        )

        if cond:
            cols.append(col)

    #combinations produce a pair only once (AB, not AB + BA)
    for col1, col2 in combinations(cols, 2):
        
        #1:many relationship exists for one of two columns
        if (( 
                df.groupby(col1)[col2].nunique().max() == 1 and
                df.groupby(col2)[col1].nunique().max() > 1
            )
        or ( 
                df.groupby(col1)[col2].nunique().max() > 1 and
                df.groupby(col2)[col1].nunique().max() == 1
            )):
            
        #ancestor (1 in 1:many pair) is appened first
            if df.groupby(col1)[col2].nunique().max() > 1:
                linked.append((col1, col2))
                
            else:
                linked.append((col2, col1))

    return linked

def find_pair_linked_columns(df):
    '''
    Given a dataframe df, return a list
    of tuples with column names where each value in 
    one column is always paired with the 
    same value in another.

    Returns a list of lists where the first column
    in the tuple is the reference one that has the weights 
    and whose parameter values cascade down to other
    linked columns.

    Currently, reference column is decided based on
    the length of its values; codes and other identifiers
    tend to have shorter values.
    '''

    linked = []
    
    #single value & numeric columns are ignored
    cols = [col for col in df.columns if
    df[col].nunique() > 1 and col not in df.select_dtypes(include=np.number)
    ]
    
    #combinations produce a pair only once (AB, not AB + BA)
    for col1, col2 in combinations(cols, 2):
        
        if ( 
                df.groupby(col1)[col2].nunique().max() == 1 and
                df.groupby(col2)[col1].nunique().max() == 1
            ):
            #column with a higher average value length is appended first
            #so that codes are paired with descriptions
            if (
                sum(map(len, df[col1].astype(str).unique())) / df[col1].nunique() >
                sum(map(len, df[col2].astype(str).unique())) / df[col2].nunique()
            ):

                linked.append([col1, col2])

            else:

                linked.append([col2, col1])

    return _merge_common_member_tuples(linked)

# INNER MODULE METHODS & CLASSES
# ==============================
class _CustomDict(defaultdict):
    '''
    Merging tuples involves sets which in turn involves arbitrary
    sort order, breaking the parent-child relationships of the
    initial pairs (Location Code, Location Desc).

    Here we monkey-patch default dict so that on encountering a missing
    key, the factory can make use of that key and create a custom 
    ordering list allowing us to recall the original sort order of the tuples
    after they have been merged.

    Key-value pairs look like {"A": [n, "A"]} where n is the "A"s position
    in the pecking order of columns.  
    '''
    
    def __init__(self, f_of_x):
        super().__init__(None) # base class doesn't get a factory
        self.f_of_x = f_of_x # save f(x)
    def __missing__(self, key): # called when a default needed
        ret = self.f_of_x(key) # calculate default value
        self[key] = ret # and install it in the dict
        return ret

class _LinkedDataGenerator:
    '''
    Generating data for linked columns is more challenging because
    columns in the same linked group can follow different rules,
    depending on the number of unique values and their anonymising
    pattern.
    '''

    def __init__(self, spec_dict, linked_group: Tuple[int, List[str]], num_rows):

        self.spec_dict = spec_dict
        self.linked_group = linked_group
        self.linked_cols = linked_group[1]
        #take the root name of the set (mountains in case of mountains.peak)
        self.anon_set = (
            spec_dict['columns'][self.linked_cols[0]]['anonymising_set'].split(".")[0])

        self.id = self.spec_dict['metadata']['id']
        self.num_rows = num_rows
        self.base_col = None
        self.base_col_pos = None
        self.scenario = None
        self.base_col_unique_count = None
        self.table_name = None
        self.sql_df = None
        self.linked_df = None

        #find the FIRST "base_col" with weights, starting from the end of the list
        #weights and probabilities are only there for columns whose unique count <= ct
        ct = spec_dict['metadata']['category_threshold']

        for i, col_name in enumerate(reversed(self.linked_cols)):
            if spec_dict['columns'][col_name]['uniques'] <= ct:
                self.base_col = col_name
                self.base_col_pos = len(self.linked_cols) - (i + 1)
                self.base_col_unique_count = spec_dict['columns'][col_name]['uniques']
                break
    
        #if ALL columns in the linked group have more unique values than allowed,
        #generate uniform distribution from the most granular and do upstream lookup
        if not self.base_col:
            self.base_col = self.linked_cols[-1]
            self.base_col_pos = -1
            self.base_col_unique_count = spec_dict['columns'][self.base_col]['uniques']
            self.scenario = 1

        elif self.base_col == self.linked_cols[-1]:
            self.scenario = 3
        else:
            self.scenario = 2
            
        #all relevant linked data is pulled from SQL into sql_df attribute
        self.sql_df = self.build_sql_dataframe()

    def build_sql_dataframe(self):
        '''
        Values for linked columns can be drawn either from the original values stored 
        in the linked group's table or they can be drawn from a pre-defined set, like
        mountaints or patients. You can also specify that only certain columns are 
        extracted, like CHI number and age, even though the full patients set contains
        other columns like first and last names, etc.
        '''

        if self.anon_set != "random":
            #either all linked columns are the same (mountains) or
            #each column is explicitly mapped to a SQL table column
            #like mountains.range and mountaints.peak. Validator will
            #error out when linked columns don't share the root table
            #or when there is a mix of notations.

            sql_df = query_anon_database(self.anon_set)
            filter_cols = []

            for col in self.linked_cols:

                col_anon_set = self.spec_dict["columns"][col]["anonymising_set"]
                
                #spec column has a dot notation to reference a specific SQL column
                if not col_anon_set == self.anon_set:
                    filter_cols.append(col_anon_set.split(".")[1])
            
            if filter_cols:
                sql_df = sql_df[filter_cols]
  
            #rename SQL columns to linked_group cols
            sql_df.columns = self.linked_cols

        else:
            #column names match the spec
            table_name = f"temp_{self.id}_{self.linked_group[0]}"
            sql_df = query_anon_database(table_name)

        return sql_df
        

    def pick_scenario(self):
        '''
        Code path resolver for linked data generation.

        Remember that all linked columns are stored in SQLite anon.db!

        Currently, there are three scenarios (each with two flavours: random & aliased)
          - values in all linked columns are drawn from uniform distribution
                This happens when the number of unique values in each column
                exceeds the user-specified threshold. In this case, the values
                are stored in anon.db and the user has no way to specify bespoke
                probabilities.

          - there ARE user-defined probabilities for ONE of the linked columns,
            but it's not the most granular column in the group, like NHS Board
            in the NHS Board + Hospital linked group.
                For this scenario, we need to respect the probabilities of the 
                given column and only draw from uniform distribution AFTER we
                generated the probability-driven values of the base column.
        
          - The most granular column in the group has user-defined probabilities,
            like Hospital in the NHS Board + Hospital linked group.
                This scenario is the easiest one because once we generate values
                for the base column, the rest of the linked columns can be derived

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame with linked data

        Because of multiple return points inside each scenario,
        it's cleaner to add paired columns in this portion of
        the code.
        '''

        if self.scenario == 1:
            linked_df = self.scenario_1()
            result = self.add_paired_columns(linked_df)

            return result

        if self.scenario == 2:
            linked_df = self.scenario_2()
            linked_df = self.alias_linked_column_values(linked_df)
            result = self.add_paired_columns(linked_df)

            return result

        if self.scenario == 3:
            linked_df = self.scenario_3()
            linked_df = self.alias_linked_column_values(linked_df)
            result = self.add_paired_columns(linked_df)

            return result
        
        return None # pragma: no cover

    def alias_linked_column_values(self, linked_df):
        '''
        If anonymising set is random and a linked column has "original values"
        in the spec, we must respect any user-made changes to values. The linked
        dataframe is still generated using original values, but if those are changed
        by the user, we will alias the originals to match.

        Original_values are in sorted order (except for Missing data which is always
        last), which means that we can sort the original linked values from anon.db
        the same way and map the values in LINKED_DF (which might or might not have
        ALL of original values due to probability vectors) to the user-edited ones
        from the spec (and not from the DB as would be the case normally)

        There is an edge case of when user deletes or adds a row in the spec which
        would mean the number of "aliases" won't match the number of "originals" put
        and then extracted from anon.db. 

        Make changes in-place
        '''
        # we need original values to act as reference and self.sql_df can be mountains
        # or other aliased dataset.        
        linked_table_name = f"temp_{self.id}_{self.linked_group[0]}"

        for linked_col in self.linked_cols: #noqa

            anon_set = self.spec_dict['columns'][linked_col]['anonymising_set']
            orig_vals = self.spec_dict['columns'][linked_col]['original_values']
            
            if anon_set == 'random' and isinstance(orig_vals, pd.DataFrame):

                original_col_values = sorted(
                    query_anon_database(
                        table_name=linked_table_name,
                        column=linked_col.replace(" ", "$")
                    )[linked_col])
                
                #potentially, user-edited
                current_col_values = orig_vals[linked_col]
                # Missing data is always last in orig_vals, but not always in anon.db
                if "Missing data" in original_col_values:
                    original_col_values.append(
                        original_col_values.pop(
                            original_col_values.index("Missing data")
                        )
                    )
                else:
                    original_col_values.append("Missing data")

                repl_dict = dict(
                    zip(
                        original_col_values,
                        current_col_values
                    )
                )

                linked_df[linked_col] = linked_df[linked_col].map(repl_dict)

        return linked_df

    def scenario_1(self):
        '''
        Values in all linked columns are drawn from a uniform distribution
        '''

        idx = np.random.choice(len(self.sql_df), self.num_rows)

        anon_list = [self.sql_df.iloc[x, :].values for x in idx]

        linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)

        return linked_df

    def scenario_2(self):
        '''
        There ARE user-defined probabilities for ONE of the linked columns,
        but it's not the most granular column in the group. Remember, that
        this column will be the first FROM THE END of the linked columns group.

        To make things simpler, we don't generate non-user defined columns
        individually, instead we generate all of them at once, using the indices
        and sizes of the base column's probabilistically drawn values.
        '''

        if self.anon_set != "random":
            #replace original_values with anonymised aliases for weights_table
            #except for the Missing data which is a special value and is always last
            #we only do it for the column with actual probabilities / weights, not
            #the child columns which won't have continuous column weights.

            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            repl = self.sql_df[self.base_col].unique()[0:self.base_col_unique_count]
            aliased_df = orig_df.replace(orig_df[self.base_col].values[:-1], repl)
            self.spec_dict['columns'][self.base_col]['original_values'] = aliased_df

        #process the first (base) parent column
        base_col_df = self.spec_dict['columns'][self.base_col]['original_values']
        base_col_prob = np.array(base_col_df['probability_vector'])
        base_col_prob /= base_col_prob.sum()

        base_col_series = pd.Series(
            data=np.random.choice(
                a=base_col_df[self.base_col].unique(),
                size=self.num_rows,
                p=base_col_prob),
            name=self.base_col   
        )

        # once we've satisfied the probabilities of the base column,
        # all we need to do is to generate random uniform indices of
        # rows to match the generated values, sized accordingly.
        base_col_counts = base_col_series.value_counts()

        sub_dfs = []

        for base_col_value, size in base_col_counts.iteritems():
            
            pool_of_idx = (
                self.sql_df[self.sql_df[self.base_col] == base_col_value].index)
            rnd_idx = np.random.choice(a=pool_of_idx, size=size)
            sub_dfs.append(
                self.sql_df[self.linked_cols].iloc[rnd_idx])
        
        result = pd.concat(sub_dfs).reset_index(drop=True)

        return result       

    def scenario_3(self):
        '''
        base_col has original_values, AND it's the most granular column
        '''

        if self.anon_set != "random":

            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            repl = self.sql_df[self.base_col].unique()[0:self.base_col_unique_count]
            aliased_df = orig_df.replace(orig_df[self.base_col].values[:-1], repl)
            self.spec_dict['columns'][self.base_col]['original_values'] = aliased_df

        #whether aliased or not
        base_col_df = self.spec_dict['columns'][self.base_col]['original_values']
        base_col_prob = np.array(base_col_df['probability_vector'])
        base_col_prob /= base_col_prob.sum()

        base_col_series = pd.Series(
            data=np.random.choice(
                a=base_col_df[self.base_col].unique(),
                size=self.num_rows,
                p=base_col_prob),
            name=self.base_col   
        )

        missing_data_row = ("Missing data",) * len(self.linked_cols)
        self.sql_df.loc[len(self.sql_df) + 1] = missing_data_row

        #join all left-side columns to base_col_series
        linked_df = pd.merge(
                left=base_col_series,
                right=self.sql_df.drop_duplicates(),
                how="left",
                on=self.base_col
            )
        
        return linked_df

    def add_paired_columns(self, linked_df):
        '''
        Doc string
        '''

        if self.anon_set != "random":

            for c in self.linked_cols:
                
                #just generate a DF with duplicate paired columns
                for pair in self.spec_dict['columns'][c]['paired_columns']:

                    #overwrite linked_df
                    linked_df = pd.concat(
                        [linked_df, pd.Series(linked_df[c], name=pair)],
                        axis=1
                    )

            return linked_df

        #if anonimysing set IS random
        for c in self.linked_cols:

            paired_columns_lookup = _create_paired_columns_lookup(self.spec_dict, c)

            if not paired_columns_lookup is None:

                linked_df = pd.merge(
                    left=linked_df,
                    right=paired_columns_lookup,
                    how="left",
                    on=c)

        return linked_df

def _merge_common_member_tuples(paired_tuples):
    '''
    Merge tuples while preserving sort order
    (codes are paired with descriptions, not the other way round)

    The position in the original paried_tuples is important: 
    it's based on the average length of all values in each column
    so in (Description, Code), Description is meant to go first

    Parameters
    ----------
    paired_tuples : list
        list of pairs of 1:1 linked columns
    
    Returns
    -------
    A list of tuples where common members have been merged into
    a single new tuple:
    Given [("A","B"), ("B","C"), ("D","E")] we want
    [["A", "B", "C"], ["D", "E"]]
    '''

    original_sort_order = _CustomDict(lambda x: [None, x])
    for a, b in paired_tuples:

        if not original_sort_order[a][0]:
            original_sort_order[a][0] = 0
        else:
            original_sort_order[a][0] -= 1

        if not original_sort_order[b][0]:
            original_sort_order[b][0] = 1
        else:
            original_sort_order[b][0] -= 0

    l = sorted(paired_tuples, key=min)
    queue = deque(l)

    grouped = []
    while len(queue) >= 2:
        l1 = queue.popleft()
        l2 = queue.popleft()
        s1 = set(l1)
        s2 = set(l2)

        if s1 & s2:
            queue.appendleft(s1 | s2)
        else:
            grouped.append(s1)
            queue.appendleft(s2)
    if queue:
        grouped.append(queue.pop())

    #convert back to mutable lists so that we can access indices
    result = [list(sorted(x, key=lambda x: original_sort_order[x])) for x in grouped]
    return result

def _create_paired_columns_lookup(spec_dict, base_column):
    '''
    Paired columns can either be in SQL or in original_values linked to base_column
    
    Parameters
    ----------
    spec_dict : dict
        the usual
    base_column : str
        column to check for presence of any paired columns

    Returns
    -------
    A dataframe with base column and paired columns, if any.
    Paired columns are stripped of their "paired_" prefix and
    the $ replacement for joining downstream into the final
    anonymised dataframe
    '''

    #get a list of paired columns:
    pairs = spec_dict['columns'][base_column]['paired_columns']
    #sanitse base_columns name for SQL
    safe_base_col_name = base_column.replace(" ", "$")

    table_name = f"temp_{spec_dict['metadata']['id']}_{safe_base_col_name}"

    if pairs:
        #check if paired column values live in SQL or are part of original_values
        if exceeds_ct(spec_dict, base_column):

            paired_df = query_anon_database(table_name=table_name)
            paired_df.rename(columns=lambda x: x.replace('paired_', ''), inplace=True)
            paired_df.rename(columns=lambda x: x.replace('$', ' '), inplace=True)
            #Missing data isn't ever in the SQL - but if Missing data is generated for
            #one of the linked columns, we want to propagate it to its links
            paired_df.loc[len(paired_df) + 1] = ["Missing data"] * paired_df.shape[1]

            return paired_df

        #code to pull the base_column + paired column(s) from original_values
        base_df = spec_dict['columns'][base_column]['original_values']

        paired_df = (
            base_df[[base_column] + [f"paired_{x}" for x in pairs]]
                .rename(columns=lambda x: x.replace('paired_', ''))
        )
        
        return paired_df
                            
    return None #pragma: no cover
