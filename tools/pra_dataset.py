"""Functions that help us setup PRA to run on the datasets.

   Outputs from these function will be located in a folder named `pra` under
   the directory of the respective dataset.
"""

import os
import numpy as np
import pandas as pd


def create_graph_input(dataset_dirpath, names_fname=['train.txt', 'test.txt', 'valid.txt'],
                       order=['head', 'relation', 'tail'], sep='\t', skiprows=0,
                       labels=['valid.txt', 'test.txt'], extension='.txt'):
    """Creates a `xxx.tsv` files that contain a positive observation for each row in the order
    (head, relation, tail), tab-separated. Takes as input `xxx.txt` files, tab-separated, where
    each row represents an observation with columns in the same order, that can have a 4th column
    that corresponds to the label (-1 or 1).

    Arguments:
    - dataset_dirpath: path to the dataset directory where all files should be in.
    - names_fpath: should be a list of file names (e.g., train.txt, test.txt, valid.txt).
    - labels: indicates which files have labels in the last row, so the function know what is the
    correct thing to do.
    """
    # ensure the `pra` directory exists
    pra_input_dir = dataset_dirpath + '/pra_graph_input/'
    if not os.path.exists(pra_input_dir):
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
        pos_df.to_csv(pra_input_dir + new_fname, columns=['head', 'relation', 'tail'],
                      index=False, header=False, sep='\t')


## ideally, these files will be placed under the directory of a specific (trained) model,
## to separate different predictions
def create_split(datasets_dirpath, split_dirpath, folds=['train.tsv', 'valid.tsv', 'test.tsv'],
                 extension='.tsv'):
    """Creates a split directory that PRA algorithm can use for the respective dataset.

    Arguments:
    - datasets_dirpath: path to the dataset directory where all files should be in
    - folds: a list of folds (data sets) (e.g., train.tsv, test.tsv, valid.tsv). Each fold should
    be written in the following order: [head, relation, tail, label], tab-separated.
    """
    # read datasets into dfs
    dfs = {}
    for fold in folds:
        fold_name = fold.replace(extension, '')
        path = datasets_dirpath + fold
        dfs[fold_name] = pd.read_csv(path, sep='\t', skiprows=0,
                                     names=['head', 'relation', 'tail', 'label'])

    # get relations
    rels = set()
    for _, df in dfs.iteritems():
        rels.add(df['relation'].unique())

    # create relations_to_run.tsv file
    with open(split_dirpath + '/relations_to_run.tsv', 'w') as f:
        for rel in rels:
            f.write(rel + '\n')

    # create each relation dir and its files
    for rel in rels:
        for fold_name, df in dfs.iteritems():
            filtered = df.loc[df['relation'] == rel]
            filtered.to_csv(split_dirpath + '/' + rel + '/' + fold_name + '.tsv',
                            columns=['head', 'tail', 'label'], index=False, header=False, sep='\t')
