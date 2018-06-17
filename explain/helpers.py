from __future__ import division
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
            ent_pair, label, features = line.replace('\n', '').split('\t')
            head, tail = ent_pair.split(',')
            d = {}
            if features:
                for feat in features.split(' -#- '):
                    feat_name, value = feat.split(',')
                    d[feat_name] = float(value)

            heads.append(head)
            tails.append(tail)
            labels.append(int(label))
            feat_dicts.append(d)

    return np.array(heads), np.array(tails), np.array(labels), feat_dicts


def getattr_else_None(class_, attr_name):
    try:
        attr = getattr(class_, attr_name)
    except AttributeError:
        attr = None
    return attr


def get_dirs(dirpath):
    """Same as `os.listdir()` but ensures that only directories will be returned.
    """
    dirs = []
    for f in os.listdir(dirpath):
        f_path = os.path.join(dirpath, f)
        if os.path.isdir(f_path) and f[0] != '.':
            dirs.append(f)
    return dirs


def ensure_dir(dirpath):
    """Creates the directory if it does not exist.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def ensure_parentdir(filepath):
    """Creates the parent directories of a file if they do not exist.
    """
    dir_ = os.path.dirname(filepath)
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def get_reasons(row, n=10):
    # Remove zero elements
    reasons = row[row != 0]
    # Select the top n_examples elements
    top_reasons_abs = reasons.abs().nlargest(n=n, keep='first')
    # Create a pandas series with these
    output = pd.Series()
    counter = 1
    for reason, _ in top_reasons_abs.iteritems():
        output['reason' + str(counter)] = reason
        output['relevance' + str(counter)] = reasons[reason]
        counter = counter + 1
        if counter == n:
            break
    for i in range(counter, n):
        output['reason' + str(i)] = "n/a"
        output['relevance' + str(i)] = "n/a"
    return output


def get_metrics(dataframe_path, model_info, split):
    relevant_test_metrics = ['Test Accuracy',
                             'Test Embedding Accuracy',
                             'Test F1_score',
                             'Test Positive Ratio',
                             'Test Precision',
                             'Test Recall',
                             'True Test Accuracy',
                             'True Test F1_score',
                             'True Test Positive Ratio',
                             'True Test Precision',
                             'True Test Recall',
                             'Test # Relevant Features / Example',
                             'Test Accuracy (Weighted # Features)',
                             'True Test Accuracy (Weighted # Features)',
                             'Test Accuracy (Weighted # Relevant Features)',
                             'True Test Accuracy (Weighted # Relevant Features)',
                             'Test # Features / Example',
                             'Test # Relevant Features / Example',
                             'Test % Examples with # Features > 0',
                             'Test % Examples with # Relevant Features > 0',
                             ]

    relevant_train_metrics = ['Train Accuracy',
                              'Train Embedding Accuracy',
                              'Train F1_score',
                              'Train Positive Ratio',
                              'Train Precision',
                              'Train Recall',
                              'True Train Accuracy',
                              'True Train F1_score',
                              'True Train Positive Ratio',
                              'True Train Precision',
                              'True Train Recall',
                              'Train # Relevant Features / Example',
                              'Train Accuracy (Weighted # Features)',
                              'True Train Accuracy (Weighted # Features)',
                              'Train Accuracy (Weighted # Relevant Features)',
                              'True Train Accuracy (Weighted # Relevant Features)',
                              'Train # Features / Example',
                              'Train # Relevant Features / Example',
                              'Train % Examples with # Features > 0',
                              'Train % Examples with # Relevant Features > 0',
                              ]

    relevant_total_metrics = ['Total # Features / Example',
                              'Total # Relevant Features / Example',
                              'Total % Examples with # Features > 0',
                              'Total % Examples with # Relevant Features > 0',
                              ]

    relevant_test_metrics_filtered = ['Test Accuracy (Filter # Features > 0)',
                                      'True Test Accuracy (Filter # Features > 0)',
                                      ]
    relevant_train_metrics_filtered = ['Train Accuracy (Filter # Features > 0)',
                                       'True Train Accuracy (Filter # Features > 0)',
                                       ]
    relevant_total_metrics_filtered = []

    relevant_test_metrics_filtered_relevant = ['Test Accuracy (Filter # Relevant Features > 0)',
                                               'True Test Accuracy (Filter # Relevant Features > 0)',
                                               'Test Mean Rule Length (Filter # Relevant Features > 0)',
                                               ]
    relevant_train_metrics_filtered_relevant = ['Train Accuracy (Filter # Relevant Features > 0)',
                                                'True Train Accuracy (Filter # Relevant Features > 0)',
                                                'Train Mean Rule Length (Filter # Relevant Features > 0)',
                                                ]
    relevant_total_metrics_filtered_relevant = []

    per_relation_metrics = pd.read_csv(dataframe_path, sep='\t')
    sum_train = per_relation_metrics['# Triples Train'].sum()
    sum_test  = per_relation_metrics['# Triples Test '].sum()
    sum_total = sum_train + sum_test
    per_relation_metrics['train_weights'] = per_relation_metrics['# Triples Train']/sum_train
    per_relation_metrics['test_weights'] = per_relation_metrics['# Triples Test ']/sum_test
    per_relation_metrics['total_weights'] = (per_relation_metrics['# Triples Test '] + per_relation_metrics['# Triples Train'])/sum_total
    # same as above but now filtering for examples with at least one feature
    sum_train_filtered = per_relation_metrics['# Triples Train (Filter # Features > 0)'].sum()
    sum_test_filtered  = per_relation_metrics['# Triples Test  (Filter # Features > 0)'].sum()
    sum_total_filtered = sum_train_filtered + sum_test_filtered
    per_relation_metrics['train_weights_filtered'] = per_relation_metrics['# Triples Train (Filter # Features > 0)']/sum_train_filtered
    per_relation_metrics['test_weights_filtered'] = per_relation_metrics['# Triples Test  (Filter # Features > 0)']/sum_test_filtered
    per_relation_metrics['total_weights_filtered'] = (per_relation_metrics['# Triples Test  (Filter # Features > 0)'] + per_relation_metrics['# Triples Train (Filter # Features > 0)'])/sum_total_filtered
    # same as above but now filtering for examples with at least one RELEVANT feature
    sum_train_filtered_relevant = per_relation_metrics['# Triples Train (Filter # Relevant Features > 0)'].sum()
    sum_test_filtered_relevant  = per_relation_metrics['# Triples Test  (Filter # Relevant Features > 0)'].sum()
    sum_total_filtered_relevant = sum_train_filtered_relevant + sum_test_filtered_relevant
    per_relation_metrics['train_weights_filtered_relevant'] = per_relation_metrics['# Triples Train (Filter # Relevant Features > 0)']/sum_train_filtered_relevant
    per_relation_metrics['test_weights_filtered_relevant'] = per_relation_metrics['# Triples Test  (Filter # Relevant Features > 0)']/sum_test_filtered_relevant
    per_relation_metrics['total_weights_filtered_relevant'] = (per_relation_metrics['# Triples Test  (Filter # Relevant Features > 0)'] + per_relation_metrics['# Triples Train (Filter # Relevant Features > 0)'])/sum_total_filtered_relevant

    metrics_dict = {}

    # calculate averages ignoring nans (which can happen for metrics that are weighted or filtered).
    for metric in relevant_test_metrics:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['test_weights'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_train_metrics:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['train_weights'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_total_metrics:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['total_weights'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    # same as above but now for filtered metrics
    for metric in relevant_test_metrics_filtered:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['test_weights_filtered'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_train_metrics_filtered:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['train_weights_filtered'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_total_metrics_filtered:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['total_weights_filtered'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    # same as above but now for filtered metrics (RELEVANT)
    for metric in relevant_test_metrics_filtered_relevant:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['test_weights_filtered_relevant'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_train_metrics_filtered_relevant:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['train_weights_filtered_relevant'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])
    for metric in relevant_total_metrics_filtered_relevant:
        metrics_dict[metric + '_micro'] = nanaverage(per_relation_metrics[metric], weights=per_relation_metrics['total_weights_filtered_relevant'])
        metrics_dict[metric + '_macro'] = np.nanmean(per_relation_metrics[metric])

    metrics_dict['dataset_name'] = model_info['dataset_name']
    metrics_dict['model_name'] = model_info['model_name']
    metrics_dict['timestamp'] = model_info['timestamp']
    metrics_dict['split'] = split

    return metrics_dict

def nanaverage(a, weights=None):
    indices = ~np.isnan(a)
    return np.average(a[indices], weights=weights[indices])
