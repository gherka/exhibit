'''
Module isolating methods and classes to find and process linked columns
'''

# Standard library imports
from itertools import chain, combinations

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
