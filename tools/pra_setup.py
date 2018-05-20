"""Functions that help us to setup dependencies (datasets, splits, etc.) for PRA to be run.
"""

import os
import numpy as np
import pandas as pd


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_graph_input(dataset_dirpath, names_fname=['train.txt', 'test.txt', 'valid.txt'],
                       order=['head', 'relation', 'tail'], sep='\t', skiprows=0,
                       labels=['valid.txt', 'test.txt'], extension='.txt',
                       graph_input_dirname='/pra_graph_input/', use_ids=False):
    """Creates a `xxx.tsv` files that contain a positive observation for each row in the order
    (head, relation, tail), tab-separated. Takes as input `xxx.txt` files, tab-separated, where
    each row represents an observation with columns in the same order, that can have a 4th column
    that corresponds to the label (-1 or 1).

    Arguments:
    - dataset_dirpath: path to the dataset directory where all files should be in.
    - names_fpath: should be a list of file names (e.g., train.txt, test.txt, valid.txt).
    - labels: indicates which files have labels in the last column, so the function know what is
              the correct thing to do.
    """
    # ensure the `pra` directory exists
    pra_input_dir = dataset_dirpath + '/' + graph_input_dirname + '/'
    if os.path.exists(pra_input_dir):
        print("{} already exists, skipping...".format(pra_input_dir))
        return
    else:
        os.makedirs(pra_input_dir)

    # read content of names files
    for fname in names_fname:
        # check the existence of labels
        if fname in labels:
            _order = order + ['label']
        else:
            _order = order

        df = pd.read_csv(dataset_dirpath + fname, sep=sep, skiprows=skiprows, names=_order)
        # check labels again
        if fname in labels: # for the case with labels
            # separate the dataframes into positive and negative if labels exist
            pos_df = df.loc[df['label'] == 1]
            neg_df = df.loc[df['label'] == -1]
            del pos_df['label']
            del neg_df['label']
        else:
            pos_df = df

        # write file for current df in `pra` dir
        new_fname = fname.replace(extension, '.tsv')
        if use_ids: new_fname = new_fname.replace('2id', '') # remove 2id at the end of file
        pos_df.to_csv(pra_input_dir + new_fname, columns=['head', 'relation', 'tail'],
                      index=False, header=False, sep='\t')
    print("Graph input created in `{}`.".format(pra_input_dir))


## ideally, these files will be placed under the directory of a specific (trained) model,
## to separate different predictions
def create_split(dfs, splits_dirpath, split_name):
    """Creates a split directory that PRA algorithm can use for the respective dataset.

    Arguments:
    - dfs: a dict whose keys are fold names (e.g. "train", "test") and values are DataFrames with
    head, tail, relation, and label columns.
    - split_dirpath: path where the split should be created.
    """
    this_split_path = splits_dirpath + '/' + split_name
    ensure_dir(splits_dirpath)
    if not os.path.exists(this_split_path):
        os.makedirs(this_split_path)
    else:
        print('Split already exists: {}.'.format(this_split_path))
        return None
        # raise ValueError('Split {} already exists in {}.'.format(
        #         split_name, splits_dirpath))

    # get relations
    rels = set()
    for _, df in dfs.iteritems():
        rels.update(df['relation'].unique())

    # create relations_to_run.tsv file
    with open(this_split_path + '/relations_to_run.tsv', 'w') as f:
        for rel in rels:
            f.write('{}\n'.format(rel))

    # create each relation dir and its files
    for rel in rels:
        for fold_name, df in dfs.iteritems():
            relpath = '{}/{}/'.format(this_split_path, rel)
            ensure_dir(relpath)
            filtered = df.loc[df['relation'] == rel]
            filtered.to_csv('{}/{}.tsv'.format(relpath, fold_name),
                            columns=['head', 'tail', 'label'], index=False, header=False, sep='\t')
