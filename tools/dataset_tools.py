"""The goal of this file is to provide functions that help processing the
datasets."""

import pandas as pd
import numpy as np


def create_name2id_dicts_from_df(df, id_first=False):
    """Creates two dictionaries from a dataframe containing two columns: one
    for the names (e.g., entities or relations) and another for the ids.

    Arguments:
    - fpath: a pandas dataframe containing the data
    - id_first: True if ids is the first column, false otherwise

    Returns:
    Two dictionaries: one that maps names to ids and another that maps ids to
    names
    """
    name2id = {row[0]: row[1] for _, row in df.iterrows()}
    id2name = {id_: name for name, id_ in name2id.iteritems()}
    return name2id, id2name


def read_name2id_file(fpath, sep='\t', skiprows=1, id_first=False):
    """Reads a file that maps names (of entities or relations) to ids
    (numbers).

    Returns:
    Two dictionaries: one that maps names to ids and another that maps ids to
    names"""
    df = pd.read_csv(fpath, sep=sep, skiprows=skiprows)
    return create_name2id_dicts_from_df(df, id_first=id_first)


def generate_type_constrain_file(dataset_path='./', fname='/type_constrain.txt'):
    """Creates a new file named `type_constrain.txt` that contains two lines for
    each relation in the dataset. The first line contains information on the
    entities that were observed as head (first position) of the relation. The
    second one contains information on the entities that were observed as being
    in the tail (last position) of the relation. Each line contains the
    following columns:
    
    - Column 1: the relation
    - Column 2: the number of different entities that were observed in that
      position (head or tail)
    - Columns 3 onwards: each different entity observed in the respective
      position
    
    The first line of the file will contain the number of relations observed in
    the dataset.

    Arguments:
    - dataset_path: the path to where the dataset is located.
    - fname: the name of the file to be generated.
    """
    train_fpath = dataset_path + '/train2id.txt'
    valid_fpath = dataset_path + '/valid2id.txt'
    test_fpath = dataset_path + '/test2id.txt'
    sep = ' '
    names = ['e1', 'e2', 'rel']

    train = pd.read_csv(train_fpath, sep=sep, skiprows=1, names=names)
    valid = pd.read_csv(valid_fpath, sep=sep, skiprows=1, names=names)
    test = pd.read_csv(test_fpath, sep=sep, skiprows=1, names=names)

    data = pd.concat([train, valid, test])

    # create a list containing the information
    rels = data.rel.unique() # get all relations contained in the dataframe
    out_list = []
    for rel in rels:
        rel_rows = data.loc[data.rel == rel] # get only rows with the respective relation
        e1s = rel_rows.e1.unique()
        e2s = rel_rows.e2.unique()
        out_list.append(np.concatenate(([rel, len(e1s)], e1s)))
        out_list.append(np.concatenate(([rel, len(e2s)], e2s)))

    # write information to file
    with open(dataset_path + fname, 'w') as f:
        f.write(str(len(rels)))
        f.write('\n')
        for line in out_list:
            line = [str(i) for i in line] # convert everything to string
            f.write('\t'.join(line)) # tab-separate everything
            f.write('\n')
    
