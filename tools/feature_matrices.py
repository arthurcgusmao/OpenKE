import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


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
            ent_pair, label, features = line.split('\t')
            head, tail = ent_pair.split(',')
            d = {}
            for feat in features.split(' -#- '):
                feat_name, value = feat.split(',')
                d[feat_name] = value

            heads.append(head)
            tails.append(tail)
            labels.append(int(label))
            feat_dicts.append(d)
    return heads, tails, labels, feat_dicts


def is_nan(x):
    return (x is np.nan or x != x)


def parse_feature_matrix_old(filepath):
    """Returns four objects: three lists (of heads, tails and labels) and a sparse matrix (of
    features) for the input (a path to a feature matrix file).
    """
    heads = []
    tails = []
    labels = []
    feat_dicts = []
    # Check if file exists beforehand
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                ent_pair, label, features = line.split('\t')
                head, tail = ent_pair.split(',')
                d = {}
                for feat in features.split(' -#- '):
                    feat_name, value = feat.split(',')
                    d[feat_name] = value

                heads.append(head)
                tails.append(tail)
                labels.append(int(label))
                feat_dicts.append(d)
        return heads, tails, labels, feat_dicts
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN


def parse_matrices_for_relation(pra_results_dpath, relation_name):
    """Takes as input the path to a directory where PRA code outputted its results (the feature
    matrices) and the relation name for one of the relations in the problem. It returns lists of
    heads, tails, labels and a sparse matrix of features for each data fold (train, valid and test).
    """
    train_fpath = "{}/{}/train.tsv".format(pra_results_dpath, relation_name)
    valid_fpath = "{}/{}/valid.tsv".format(pra_results_dpath, relation_name)
    test_fpath = "{}/{}/test.tsv".format(pra_results_dpath, relation_name)

    train_heads, train_tails, train_labels, train_feat_dicts = parse_feature_matrix_old(train_fpath)
    valid_heads, valid_tails, valid_labels, valid_feat_dicts = parse_feature_matrix_old(valid_fpath)
    test_heads, test_tails, test_labels, test_feat_dicts = parse_feature_matrix_old(test_fpath)

    v = DictVectorizer(sparse=True)
    v.fit(train_feat_dicts)

    # Not all relations have validation and test data
    # Setting those to None by default
    valid_X, test_X = np.NaN, np.NaN
    # If they exist, transform their feature matrices
    train_X = v.transform(train_feat_dicts)
    if not is_nan(valid_feat_dicts):
        valid_X = v.transform(valid_feat_dicts)
    if not is_nan(test_feat_dicts):
        test_X = v.transform(test_feat_dicts)

    return {
        'train_heads': train_heads,
        'train_tails': train_tails,
        'train_labels': train_labels,
        'train_X': train_X,
        'valid_heads': valid_heads,
        'valid_tails': valid_tails,
        'valid_labels': valid_labels,
        'valid_X': valid_X,
        'test_heads': test_heads,
        'test_tails': test_tails,
        'test_labels': test_labels,
        'test_X': test_X,
        'feature_names': v.get_feature_names()
    }
