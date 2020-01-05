'''
Module isolating methods and classes to find, process and generate linked columns
'''

# Standard library imports
from itertools import chain, combinations
import sqlite3
from contextlib import closing

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from exhibit.core.utils import package_dir, exceeds_ct
from exhibit.core.sql import query_anon_database

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

def find_hierarchically_linked_columns(df):
    '''
    Given a dataframe df, return a list
    of tuples with column names where values in 
    the second column are always paired with the 
    same value in the first column (many:1 relationship)
    '''
    linked = []
    
    #single value columns are ignored
    cols = [col for col in df.columns if df[col].nunique() > 1]
    
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

    Need to implement a "common member" merging of tuples
    so that [(A,B), (B,C), (D,E)] gets changed into 
    [(A,B,C), (D,E)]
    '''
    linked = []
    
    #single value columns are ignored
    cols = [col for col in df.columns if df[col].nunique() > 1]
    
    #combinations produce a pair only once (AB, not AB + BA)
    for col1, col2 in combinations(cols, 2):
        
        if ( 
                df.groupby(col1)[col2].nunique().max() == 1 and
                df.groupby(col2)[col1].nunique().max() == 1
            ):
            #column with a higher average value length is appended first
            if (
                sum(map(len, df[col1].astype(str).unique())) / df[col1].nunique() >
                sum(map(len, df[col2].astype(str).unique())) / df[col2].nunique()
            ):

                linked.append([col1, col2])

            else:

                linked.append([col2, col1])

    return linked

def find_linked_columns(df):
    '''
    Given a dataframe df, return a list
    of tuples with column names where values in 
    one column are always paired with the 
    same value in another, as in, for example,
    an NHS Board and NHS Board Code.

    Columns with the same value in all rows are skipped.

    The column that maps 1 to many is appended first,
    judging by the number of unique values it has.
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


def create_paired_columns_lookup(spec_dict, base_column):
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

    if spec_dict['metadata']['id'] == 'sample':
        table_name = f"sample_{safe_base_col_name}"
    else:
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

class LinkedDataGenerator:

    def __init__(self, spec_dict, linked_group, num_rows):

        self.linked_cols = spec_dict['constraints']['linked_columns'][linked_group][1]
        self.ct = spec_dict['metadata']['category_threshold']
        self.anon_set = spec_dict['columns'][self.linked_cols[0]]['anonymising_set']
        self.spec_dict = spec_dict

        self.num_rows = num_rows
        self.base_col = None
        self.base_col_pos = None
        self.all_cols_uniform = False
        self.base_col_unique_count = None
        self.table_name = None
        self.sql_rows = None
        self.linked_df = None

        #find the first available "base_col", starting from the end of the list
        for i, col_name in enumerate(reversed(self.linked_cols)):
            if spec_dict['columns'][col_name]['uniques'] <= self.ct:
                self.base_col = list(reversed(self.linked_cols))[i]
                self.base_col_pos = i
                self.base_col_unique_count = spec_dict['columns'][col_name]['uniques']
                if self.base_col:
                    break
    
        #if all columns in the linked group have more unique values than allowed,
        #just generate uniform distribution from the most granular and do upstream lookup
        if not self.base_col:
            self.base_col = list(reversed(self.linked_cols))[0]
            self.base_col_pos = 0
            self.base_col_uniform = True
            self.base_col_unique_count = spec_dict['columns'][self.base_col]['uniques']
        
        #Generator can have two flavours: random (using existing values) and aliased
        if self.anon_set != "random":
            self.table_name = self.anon_set
            #OK to limit the size ot base col uniques because it's the most granular
            self.anon_df = query_anon_database(self.table_name, size=self.base_col_unique_count)
            #rename the first column of the anon_set df to be same as original
            self.anon_df.rename(columns={self.anon_df.columns[0]:self.base_col}, inplace=True)

        else:
            #sanitise the column name in case it has spaces in it
            base_col_sql = self.base_col.replace(" ", "$")
            
            #special case for reference test table for the prescribing dataset
            if spec_dict['metadata']['id'] == "sample":
                self.table_name = f"sample_{linked_group}"
            else:
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
        Code path resolver

        Because of multiple return points, it's cleaner
        to add paired columns in this portion of the code
        rather than in the main scenario run
        '''
        if self.all_cols_uniform:
            linked_df = self.scenario_1()
            linked_df = self.add_paired_columns(linked_df)
            return linked_df

        elif self.base_col_pos != 0:
            linked_df = self.scenario_2()
            linked_df = self.add_paired_columns(linked_df)
            return linked_df

        elif self.base_col_pos == 0:
            linked_df = self.scenario_3()
            linked_df = self.add_paired_columns(linked_df)
            return linked_df


    def scenario_1(self):
        '''
        all_cols_uniform = True
        '''

        if self.anon_set != "random":

            idx = np.random.choice(len(self.anon_df), self.num_rows)
            #to_records returns numpy records which look like tuples, but aren't
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
        base_col has original_values, but it isn't the most granular column
        '''

        if self.anon_set != "random":

            #grab the full anonymising dataset
            full_anon_df = query_anon_database(self.table_name)
            full_anon_df.rename(
                columns={full_anon_df.columns[0]:self.base_col}, inplace=True)

            #replace original_values with anonymised aliases for weights_table
            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            orig_df.iloc[:, 0] = (full_anon_df
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
            orig_df = self.spec_dict['columns'][self.base_col]['original_values']
            orig_df.iloc[:, 0] = (full_anon_df
                                    .iloc[:, 1].unique()[0:self.base_col_unique_count])
            self.spec_dict['columns'][self.base_col]['original_values'] = orig_df

            #carry on with the programme
            base_col_df = self.spec_dict['columns'][self.base_col]['original_values']

            base_col_prob = np.array(base_col_df['probability_vector'])

            base_col_prob /= base_col_prob.sum()

            idx = np.random.choice(self.base_col_unique_count, self.num_rows, p=base_col_prob)
            anon_list = [full_anon_df.iloc[x, :].values for x in idx]

            linked_df = pd.DataFrame(columns=self.linked_cols, data=anon_list)

            return linked_df
        
        #random
        base_col_df = self.spec_dict['columns'][self.base_col]['original_values']

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

                if self.spec_dict['columns'][c]['anonymising_set'] != "random":
                    #just generate a DF with duplicate paired columns
                    for pair in self.spec_dict['columns'][c]['paired_columns']:
                        
                        #overwrite linked_df
                        linked_df = pd.concat(
                            [linked_df, pd.Series(linked_df[c], name=pair)],
                            axis=1
                        )

                    continue
            
            paired_columns_lookup = create_paired_columns_lookup(self.spec_dict, c)

            if not paired_columns_lookup is None:
                linked_df = pd.merge(
                    left=linked_df,
                    right=paired_columns_lookup,
                    how="left",
                    on=c)

            return linked_df
        #if anonimysing set IS random
        for c in self.linked_cols:

            paired_columns_lookup = create_paired_columns_lookup(self.spec_dict, c)

            if not paired_columns_lookup is None:

                linked_df = pd.merge(
                    left=linked_df,
                    right=paired_columns_lookup,
                    how="left",
                    on=c)

        return linked_df
                