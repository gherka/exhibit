'''
Module isolating methods and classes to find, process and generate linked columns
'''
# Standard library imports
from itertools import chain, combinations
import sqlite3
from contextlib import closing
from collections import deque, defaultdict

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from .utils import package_dir, exceeds_ct
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

def generate_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    Doc string
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

    def __init__(self, spec_dict, linked_group, num_rows):

        self.spec_dict = spec_dict
        self.linked_cols = spec_dict['constraints']['linked_columns'][linked_group][1]
        self.anon_set = spec_dict['columns'][self.linked_cols[0]]['anonymising_set']

        self.num_rows = num_rows
        self.base_col = None
        self.base_col_pos = None
        self.all_cols_uniform = False
        self.base_col_unique_count = None
        self.table_name = None
        self.sql_rows = None
        self.linked_df = None

        #find the FIRST "base_col" with weights, starting from the end of the list
        #weights and probabilities are only there for columns whose unique count <= ct
        ct = spec_dict['metadata']['category_threshold']

        for i, col_name in enumerate(reversed(self.linked_cols)):
            if spec_dict['columns'][col_name]['uniques'] <= ct:
                self.base_col = col_name
                self.base_col_pos = i
                self.base_col_unique_count = spec_dict['columns'][col_name]['uniques']
                break
    
        #if ALL columns in the linked group have more unique values than allowed,
        #generate uniform distribution from the most granular and do upstream lookup
        if not self.base_col:
            self.base_col = list(reversed(self.linked_cols))[0]
            self.base_col_pos = 0
            self.all_cols_uniform = True
            self.base_col_unique_count = spec_dict['columns'][self.base_col]['uniques']
        
        #Generator can have two flavours: random (using existing values) and aliased
        if self.anon_set != "random":
            self.table_name = self.anon_set
            #OK to limit the size of base col uniques because it's the most granular
            self.anon_df = query_anon_database(self.table_name, size=self.base_col_unique_count)
            #rename the first column of the anon_set df to be same as original
            self.anon_df.rename(columns={self.anon_df.columns[0]:self.base_col}, inplace=True)

        else:
            #sanitise the column name in case it has spaces in it
            base_col_sql = self.base_col.replace(" ", "$")
            
            self.table_name = f"temp_{spec_dict['metadata']['id']}_{linked_group}"

            #get the linked data out for lookup purposes later
            db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"
            conn = sqlite3.connect(db_uri, uri=True)

            sql = f"""
            SELECT *
            FROM {self.table_name}
            ORDER BY {base_col_sql}
            """

            with closing(conn):
                c = conn.cursor()
                c.execute(sql)
                self.sql_rows = c.fetchall()

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
        if self.all_cols_uniform:
            linked_df = self.scenario_1()
            linked_df = self.add_paired_columns(linked_df)

            result = self.alias_linked_column_values(linked_df)
            return result

        if self.base_col_pos != 0:
            linked_df = self.scenario_2()
            linked_df = self.add_paired_columns(linked_df)

            result = self.alias_linked_column_values(linked_df)
            return result

        if self.base_col_pos == 0:
            linked_df = self.scenario_3()
            linked_df = self.add_paired_columns(linked_df)

            result = self.alias_linked_column_values(linked_df)
            return result


    def alias_linked_column_values(self, linked_df):
        '''
        If anonymising set is random and a linked column has "original values"
        in the spec, we must respect any user-made changes to values. The linked
        dataframe is still generated using original values, but if those are changed
        by the user, we will alias the originals to match.

        Make changes in-place
        '''

        for linked_group in self.spec_dict['constraints']['linked_columns']:
            for linked_col in linked_group[1]:

                anon_set = self.spec_dict['columns'][linked_col]['anonymising_set']
                orig_vals = self.spec_dict['columns'][linked_col]['original_values']
                paired_cols = self.spec_dict['columns'][linked_col]['paired_columns']

                if anon_set == 'random' and isinstance(orig_vals, pd.DataFrame):
                    #Missing data is breaking default sort order as it's always last
                    repl_dict = dict(
                        zip(
                            sorted(linked_df[linked_col].unique()),
                            sorted(orig_vals[linked_col])
                        )
                    )

                    linked_df[linked_col] = linked_df[linked_col].map(repl_dict)

                    if paired_cols:
                        for paired_col in paired_cols:

                            repl_dict = dict(
                                    zip(
                                sorted(linked_df[paired_col].unique()),
                                sorted(orig_vals["paired_"+ paired_col])
                                )
                            )

                            linked_df[paired_col] = linked_df[paired_col].map(repl_dict)

        return linked_df


    def scenario_1(self):
        '''
        Values in all linked columns are drawn from a uniform distribution
        '''

        if self.anon_set != "random":

            idx = np.random.choice(len(self.anon_df), self.num_rows)

            anon_list = [
                list(self.anon_df.itertuples(index=False, name=None))[x] for x in idx
                ]

            linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)

            return linked_df

        idx = np.random.choice(len(self.sql_rows), self.num_rows)
        anon_list = [self.sql_rows[x] for x in idx]
        linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)

        return linked_df

    def scenario_2(self):
        '''
        There ARE user-defined probabilities for ONE of the linked columns,
        but it's not the most granular column in the group.
        '''

        if self.anon_set != "random":

            #grab the full anonymising dataset
            full_anon_df = query_anon_database(self.table_name)
            full_anon_df.rename(
                columns={full_anon_df.columns[0]:self.base_col}, inplace=True)

            #replace original_values with anonymised aliases for weights_table
            #except for the Missing data which is a special value and is always last
            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            orig_df.iloc[0:-1, 0] = (full_anon_df
                                    .iloc[:, 0].unique()[0:self.base_col_unique_count])
            self.spec_dict['columns'][self.base_col]['original_values'] = orig_df

        else:
            #sql df
            full_anon_df = pd.DataFrame(columns=self.linked_cols, data=self.sql_rows)


        base_col_df = self.spec_dict['columns'][self.base_col]['original_values']

        base_col_prob = np.array(base_col_df['probability_vector'])

        base_col_prob /= base_col_prob.sum()

        base_col_series = pd.Series(
            data=np.random.choice(
                a=base_col_df.iloc[:, 0].unique(),
                size=self.num_rows,
                p=base_col_prob),
            name=self.base_col   
        )

        uniform_series = (
            base_col_series
                .groupby(base_col_series)
                .transform(
                    lambda x: np.random.choice(
                        a=(full_anon_df[full_anon_df[self.base_col] == min(x)]
                            .iloc[:, -1]),
                        size=len(x)
                    )
                ) 
            )
        
        uniform_series.name = self.linked_cols[-1]

        linked_df = pd.concat([base_col_series, uniform_series], axis=1)

        if self.anon_set != "random":

            #create a "hidden", internal key entry: "aliases" for anonymised values
            #and use them to populate the weights table instead of default values

            uniform_table = pd.DataFrame(pd.Series(
                uniform_series.unique(),
                name=uniform_series.name
            ))

            self.spec_dict['columns'][uniform_series.name]['aliases'] = uniform_table
        
        else:
            #join the remaining columns, if there are any
            if len(self.linked_cols) > 2:
                linked_df = pd.merge(
                    left=linked_df,
                    right=full_anon_df,
                    how='left',
                    on=[self.base_col, self.linked_cols[-1]]
                )

        return linked_df


    def scenario_3(self):
        '''
        base_col has original_values, AND it's the most granular column
        '''

        if self.anon_set != "random":

            #grab the full anonymising dataset
            full_anon_df = query_anon_database(self.table_name)

            #replace original_values with anonymised aliases for weights_table
            #except for the last Missing data row
            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            orig_df.iloc[0:-1, 0] = (full_anon_df
                                    .iloc[:, 1].unique()[0:self.base_col_unique_count])
            self.spec_dict['columns'][self.base_col]['original_values'] = orig_df

            #carry on with the programme
            base_col_df = (
                self.spec_dict['columns'][self.base_col]['original_values']
            )

            base_col_prob = np.array(base_col_df['probability_vector'])

            base_col_prob /= base_col_prob.sum()
            
            #add +1 for Missing data which is part of original values, but not unique count
            idx = np.random.choice(self.base_col_unique_count + 1, self.num_rows, p=base_col_prob)
            anon_list = [full_anon_df.iloc[x, :].values for x in idx]

            linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)

            return linked_df
        
        #random
        base_col_df = (
            self.spec_dict['columns'][self.base_col]['original_values']
        )

        base_col_prob = np.array(base_col_df['probability_vector'])
        base_col_prob /= base_col_prob.sum()

        idx = np.random.choice(len(self.sql_rows), self.num_rows, p=base_col_prob)
        anon_list = [self.sql_rows[x] for x in idx]

        linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)
        
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

            return paired_df

        #code to pull the base_column + paired column(s) from original_values
        base_df = spec_dict['columns'][base_column]['original_values']

        paired_df = (
            base_df[[base_column] + [f"paired_{x}" for x in pairs]]
                .rename(columns=lambda x: x.replace('paired_', ''))
        )
        
        return paired_df
                            
    #if no pairs, just return None
    return None
