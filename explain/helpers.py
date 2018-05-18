import os
import numpy as np
import pandas as pd

def parse_feature_matrix(filepath):
    """Returns four objects: three lists (of heads, tails and labels) and a sparse matrix (of
    features) for the input (a path to a feature matrix file).
    """
    heads = []
    tails = []
    labels = []
    feat_dicts = []
    with open(filepath, 'r') as f:
        for line in f:
            ent_pair, label, features = line.rstrip().split('\t')
            head, tail = ent_pair.split(',')
            d = {}
            for feat in features.split(' -#- '):
                feat_name, value = feat.split(',')
                d[feat_name] = float(value)

            heads.append(head)
            tails.append(tail)
            labels.append(int(label))
            feat_dicts.append(d)

    return np.array(heads), np.array(tails), np.array(labels), feat_dicts


def getattr_else_None(class, attr_name):
    try:
        attr = getattr(class, attr_name)
    except AttributeError:
        attr = None


def get_dirs(dirpath):
    """Same as `os.listdir()` but ensures that only directories will be returned.
    """
    dirs = []
    for f in os.listdir(dirpath):
        f_path = os.path.join(dirpath, f)
        if os.path.isdir(f_path):
            dirs.append(f)
    return dirs


def ensure_dir(dirpath):
    """Creates the directory if it does not exist.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_reasons(row, n=10):
    # Remove zero elements
    reasons = row[row != 0]
    # Select the top n_examples elements
    top_reasons_abs = reasons.abs().nlargest(n=n, keep='first')
    # Create a pandas series with these
    output = pd.Series()
    counter = 1
    for reason, _ in top_reasons_abs.iteritems():
        reason_name, _ = reason.split('=')
        output['reason' + str(counter)] = reason_name
        output['relevance' + str(counter)] = reasons[reason]
        counter = counter + 1
        if counter == n:
            break
    for i in range(counter, n):
        output['reason' + str(i)] = "n/a"
        output['relevance' + str(i)] = "n/a"
    return output
