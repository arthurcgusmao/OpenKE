"""The goal of this file is to provide functions that help processing the datasets."""

import pandas as pd
import numpy as np


def create_name2id_dicts_from_df(df, id_first=False):
    """Creates two dictionaries from a dataframe containing two columns: one for the
    names (e.g., entities or relations) and another for the ids.

    Arguments:
    - fpath: a pandas dataframe containing the data
    - id_first: True if ids is the first column, false otherwise
    """
    name2id = {row[0]: row[1] for _, row in df.iterrows()}
    id2name = {id_: name for name, id_ in name2id.iteritems()}
    return name2id, id2name


def read_name2id_file(fpath, sep='\t', skiprows=1, id_first=False):
    df = pd.read_csv(fpath, sep=sep, skiprows=skiprows)
    return create_name2id_dicts_from_df(df, id_first=id_first)
