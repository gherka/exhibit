'''
Module isolating methods and classes to find, process and generate
hierarchically (one to many) linked columns. For user-defined linked
columns where the relationships are coded in a lookup + matrix see
the matrix module.
'''

# Standard library imports
from itertools import combinations
from typing import Tuple, List
from collections import deque, defaultdict

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from ..utils import exceeds_inline_limit
from ..sql import query_anon_database
from ..constants import MISSING_DATA_STR, ORIGINAL_VALUES_PAIRED

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

        chain counter starts from 1 because zero is reserved for user-defined
        linked columns; there is only ever going to be one group of them.
        '''

        self.chain_counter = 1
        self.chains = self.process_nodes(connections)
        self.tree = list((i, l) for i, l in self.chains.items())
     
    def process_nodes(self, connections, chains=None):
        '''
        Recursively build chains from connections.

        Returns a dictionary with chain_id : chain
        '''

        #initialise empty chains dictionary
        if chains is None:
            chains = {1: []}

        #stop condition for recursion
        if not connections:
            return chains

        #create a new chain from the first pair of nodes
        chains = self.create_new_chain(connections[0], chains)

        #fill the chain with the rest of the nodes, if able  
        finished_chain, remaining_connections = self.build_chain(
            connections, chains[self.chain_counter-1])

        chains[self.chain_counter-1] = finished_chain
        
        return self.process_nodes(remaining_connections, chains)       
        
    def build_chain(self, connections, chain):
        '''
        Changes to the passed in chain are happening in-place
        because chain is a mutable list. Maybe refactor later.

        Returns remaining connection pairs in a list after the
        initial chain was finalized.
        '''

        #make a copy of connections to remove processed pairs  
        inner_loop = list(connections)

        #if variable doesn't get overridden as part of the loop,
        #it means the chain is finished and nothing can be done
        #with any of the pairs as pertains to the worked-on chain.
        reset = False

        for connection in connections:
            #connection already exists
            if self.nodes_in_chain_already(connection, chain):
                inner_loop.remove(connection)
                reset = True
                continue
            #it's possible to extend one of the existing chains
            if self.can_extend_chain(connection, chain):
                chain = self.extend_chain(connection, chain)
                inner_loop.remove(connection)
                reset = True
                continue
            #it's possible to prepend one of the existing chains
            if self.can_prepend_chain(connection, chain):
                chain = self.prepend_chain(connection, chain)
                inner_loop.remove(connection)
                reset = True
                continue
            #is the connection one of two that can be merged and new node spliced-in
            if self.splice_node(connection, connections, chain, inner_loop):
                reset = True
        
        if reset:
            return self.build_chain(inner_loop, chain)
        
        return chain, inner_loop  

    def create_new_chain(self, connection, chains):
        '''
        Each chain has to have at least two seeding nodes.

        This function uses chain_counter from the outer scope
        (defined in the __init__)

        Returns an updated chains dictionary
        '''

        opening_node, closing_node = connection

        chains[self.chain_counter] = [opening_node, closing_node]
        self.chain_counter += 1

        return chains

    def extend_chain(self, connection, chain):
        '''
        Append new_node to the end of an existing chain.
        '''

        return chain + [connection[1]]

    def prepend_chain(self, connection, chain):
        '''
        Extend the chain in other direction, adding new node to the front
        '''

        return [connection[0]] + chain

    def nodes_in_chain_already(self, connection, chain):
        '''
        Ensure we're not creating new chains from node pairs
        that are already included in other chains.
        '''

        if connection[0] in chain and connection[1] in chain:
            return True
        return False

    def can_extend_chain(self, connection, chain):
        '''
        Convenience function
        '''

        if connection[0] == chain[-1]:
            return True
        return False

    def can_prepend_chain(self, connection, chain):
        '''
        Convenience function
        '''

        if connection[1] == chain[0]:
            return True
        return False

    def splice_node(self, connection, connections, chain, inner_loop):
        '''
        Connections are pairs of columns that are hierarchically
        linked so if multiple columns are linked in a single chain,
        we need to re-consititute the chain from these paired links.

        Also, the order of the pairs in connections is not guaranteed,
        so we need to iterate until we're certain no link is missed.

        It's also crucial that we only create un-interrupted chains so
        [B,C] is a valid link in the [A,B,C,D] chain if there are also
        [A,B] AND [C,D] links. 
        '''
        
        #Example: initial chain = [A, D]
        for splice_buddy in connections:

            splice_node = None

            if connection[1] == splice_buddy[0]: # (A, B) + (B, D)
                opening_node = connection[0]
                closing_node = splice_buddy[1]
                splice_node = connection[1]

            elif connection[0] == splice_buddy[1]: # (B, D) + (A, B)
                opening_node = splice_buddy[0]
                closing_node = connection[1]
                splice_node = connection[0]

            if splice_node:

                if (
                    opening_node in chain and
                    closing_node in chain and
                    abs(chain.index(opening_node) - chain.index(closing_node)) == 1):
                    chain.insert(chain.index(opening_node) + 1, splice_node) #in-place
                    inner_loop.remove(connection)
                    return True

        return False

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

    # make sure to scramble the order and drop index to avoid problems
    # with clustered values and sorting when pd.concat-ing
    linked_df = (
        gen.pick_scenario()
            .sample(frac=1, random_state=np.random.PCG64(0))
            .reset_index(drop=True)
    )

    return linked_df

def find_hierarchically_linked_columns(df, spec, user_linked_cols=None):
    '''
    Given a dataframe df, return a list
    of tuples with column names where values in 
    the second column are always paired with the 
    same value in the first column (many:1 relationship)
    '''

    linked = []
    
    #single value and paired columns are ignored
    cols = set()

    for col in spec["metadata"]["categorical_columns"]:
        cond = (
            (df[col].nunique() > 1) &
            (spec["columns"][col]["original_values"] != ORIGINAL_VALUES_PAIRED)
        )

        if cond:
            cols.add(col)

    # user defined linked columns take precedence over hierarchical
    if user_linked_cols is not None:
        cols = cols.difference(user_linked_cols)

    # since set is unordered, the linked tree can change from one run to another
    cols = list(sorted(cols))

    #combinations produce a pair only once (AB, not AB + BA)
    for col1, col2 in combinations(cols, 2):
        #drop NAs because replacing them with Missing data means
        #that columns that are normally linked, won't be (Missing data will
        #appear for multiple "parent" columns) 
        pair_df = df[[col1, col2]].dropna()

        #check again if after dropping NAs, the result is a single value column
        #even though there is a test that fails without this line, 286 is still
        #reported as not being covered so there is a pragma to get 100% coverage
        if pair_df[col1].nunique() == 1 or pair_df[col2].nunique() == 1:
            continue # pragma: no cover
        
        #1:many relationship exists for one of two columns
        if (( 
                pair_df.groupby(col1)[col2].nunique().max() == 1 and
                pair_df.groupby(col2)[col1].nunique().max() > 1
            )
        or ( 
                pair_df.groupby(col1)[col2].nunique().max() > 1 and
                pair_df.groupby(col2)[col1].nunique().max() == 1
            )):
            
        #ancestor (1 in 1:many pair) is appened first
            if pair_df.groupby(col1)[col2].nunique().max() > 1:
                linked.append((col1, col2))
                
            else:
                linked.append((col2, col1))

    return linked

def find_pair_linked_columns(df, ignore_cols=None):
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
    all_cols = set(df.columns)

    if ignore_cols:
        all_cols = all_cols - set(ignore_cols)
    
    #single value & numeric columns are ignored
    cols = [col for col in all_cols if
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
        self.rng = spec_dict["_rng"]
        self.linked_group = linked_group
        self.linked_cols = linked_group[1]
        #take the root name of the set (mountains in case of mountains.peak)
        self.anon_set = (
            spec_dict["columns"][self.linked_cols[0]]["anonymising_set"].split(".")[0])

        self.id = self.spec_dict["metadata"]["id"]
        self.num_rows = num_rows
        self.base_col = None
        self.base_col_pos = None
        self.scenario = None
        self.base_col_unique_count = None
        self.table_name = None
        self.sql_df = None
        self.linked_df = None

        #find the FIRST "base_col" with weights, starting from the end of the list
        #weights and probabilities are only there for columns whose
        #unique count <= inline limit
        inline_limit = spec_dict["metadata"]["inline_limit"]

        for i, col_name in enumerate(reversed(self.linked_cols)):
            if spec_dict["columns"][col_name]["uniques"] <= inline_limit:
                self.base_col = col_name
                self.base_col_pos = len(self.linked_cols) - (i + 1)
                self.base_col_unique_count = spec_dict["columns"][col_name]["uniques"]
                break
    
        #if ALL columns in the linked group have more unique values than allowed,
        #generate uniform distribution from the most granular and do upstream lookup
        if not self.base_col:
            self.base_col = self.linked_cols[-1]
            self.base_col_pos = -1
            self.base_col_unique_count = spec_dict["columns"][self.base_col]["uniques"]
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
                if col_anon_set != self.anon_set:
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

            anon_set = self.spec_dict["columns"][linked_col]["anonymising_set"]
            orig_vals = self.spec_dict["columns"][linked_col]["original_values"]
            
            if anon_set == "random" and isinstance(orig_vals, pd.DataFrame):
                
                #we need to drop missing data placeholder prior to sorting to
                #avoid it messing up the aliasing mappings which rely on two sets
                #of column names being in the same order.
                original_col_values = sorted(
                    query_anon_database(
                        table_name=linked_table_name,
                        column=linked_col.replace(" ", "$"),
                        exclude_missing=True
                    )[linked_col])
                
                #potentially, user-edited; Missind data always last
                current_col_values = orig_vals[linked_col][:-1]

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
        Values in all linked columns are drawn from a uniform distribution,
        excluding Missing data which is in SQL DB, last. Potentially need to
        pop it and reinsert since order isn't always guaranteed.
        '''

        idx = self.rng.choice(len(self.sql_df) - 1, self.num_rows)

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

        base_col_vals = None
        base_col_df = self.spec_dict["columns"][self.base_col]["original_values"][:-1]
        base_col_prob = np.array(base_col_df["probability_vector"]).astype(float)
        
        if base_col_prob.sum() != 1:
            base_col_prob /= base_col_prob.sum()

        if self.anon_set != "random":
            #replace original_values with anonymised aliases for weights_table
            #except for the Missing data which is a special value and is always last
            #we only do it for the column with actual probabilities / weights, not
            #the child columns which won't have continuous column weights.

            orig_df = self.spec_dict["columns"][self.base_col]["original_values"]
            repl = self.sql_df[self.base_col].unique()[0:self.base_col_unique_count]
            aliases = dict(zip(orig_df[self.base_col].values[:-1], repl))
            aliased_df = orig_df.applymap(lambda x: aliases.get(x, x))
            self.spec_dict["columns"][self.base_col]["original_values"] = aliased_df
            base_col_vals = aliased_df[self.base_col].iloc[:-1].unique()

        # original values are only in SQL; spec might have been modified
        if base_col_vals is None:
            base_col_vals = (
                self.sql_df[self.base_col]
                .loc[lambda x: x != MISSING_DATA_STR]
                .sort_values()
                .unique()[0:self.base_col_unique_count])

        base_col_series = pd.Series(
            data=self.rng.choice(
                a=base_col_vals,
                size=self.num_rows,
                p=base_col_prob),
            name=self.base_col   
        )

        # once we've satisfied the probabilities of the base column,
        # all we need to do is to generate random uniform indices of
        # rows to match the generated values, sized accordingly.

        # make sure we sorted the value_counts to keep RNG consistent
        base_col_counts = base_col_series.value_counts().sort_index()

        sub_dfs = []

        for base_col_value, size in base_col_counts.iteritems():
            
            pool_of_idx = (
                self.sql_df[self.sql_df[self.base_col] == base_col_value].index)
            rnd_idx = self.rng.choice(a=pool_of_idx, size=size)
            sub_dfs.append(
                self.sql_df[self.linked_cols].iloc[rnd_idx])
        
        result = pd.concat(sub_dfs)

        return result       

    def scenario_3(self):
        '''
        base_col has original_values, AND it's the most granular column
        Note that if you delete linked column values from spec, the code
        will still run, but the aliases will be taken from the top - which
        might not be desirable if certain values have distinct meaning like
        "other locations" or "no readmission". Should probably issue a warning. 
        '''

        base_col_vals = None
        base_col_df = self.spec_dict["columns"][self.base_col]["original_values"][:-1]
        base_col_prob = np.array(base_col_df["probability_vector"]).astype(float)

        if base_col_prob.sum() != 1:
            base_col_prob /= base_col_prob.sum()

        if self.anon_set != "random":

            orig_df = self.spec_dict["columns"][self.base_col]["original_values"]
            repl = self.sql_df[self.base_col].unique()[0:self.base_col_unique_count]
            aliases = dict(zip(orig_df[self.base_col].values[:-1], repl))
            aliased_df = orig_df.applymap(lambda x: aliases.get(x, x))
            self.spec_dict["columns"][self.base_col]["original_values"] = aliased_df
            base_col_vals = aliased_df[self.base_col].iloc[:-1].unique()

        # original values are only in SQL; spec might have been modified
        # it's important to sort to make sure values align
        if base_col_vals is None:
            base_col_vals = (
                self.sql_df[self.base_col]
                .loc[lambda x: x != MISSING_DATA_STR]
                .sort_values()
                .unique()[0:self.base_col_unique_count])


        base_col_series = pd.Series(
            data=self.rng.choice(
                a=base_col_vals,
                size=self.num_rows,
                p=base_col_prob),
            name=self.base_col   
        ) 

        #join all left-side columns to base_col_series
        #WILL PRODUCE NULLS IF BASE_COL_SERIES IS ALIASED IN THE SPEC!
        linked_df = pd.merge(
                left=base_col_series,
                right=self.sql_df.drop_duplicates(subset=[self.base_col], keep="last"),
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
                for pair in self.spec_dict["columns"][c]["paired_columns"] or list():

                    #overwrite linked_df
                    linked_df = pd.concat(
                        [linked_df, pd.Series(linked_df[c], name=pair)],
                        axis=1
                    )

            return linked_df

        #if anonimysing set IS random
        for c in self.linked_cols:

            paired_columns_lookup = _create_paired_columns_lookup(self.spec_dict, c)

            if paired_columns_lookup is not None:

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
    pairs = spec_dict["columns"][base_column]["paired_columns"]
    #sanitse base_columns name for SQL
    safe_base_col_name = base_column.replace(" ", "$")

    table_name = f"temp_{spec_dict['metadata']['id']}_{safe_base_col_name}"

    if pairs:
        #check if paired column values live in SQL or are part of original_values
        if exceeds_inline_limit(spec_dict, base_column):

            paired_df = query_anon_database(table_name=table_name)
            paired_df.rename(columns=lambda x: x.replace("paired_", ""), inplace=True)
            paired_df.rename(columns=lambda x: x.replace("$", " "), inplace=True)

            return paired_df

        #code to pull the base_column + paired column(s) from original_values
        base_df = spec_dict["columns"][base_column]["original_values"]

        paired_df = (
            base_df[[base_column] + [f"paired_{x}" for x in pairs]]
                .rename(columns=lambda x: x.replace("paired_", ""))
        )
        
        return paired_df
                            
    return None #pragma: no cover
