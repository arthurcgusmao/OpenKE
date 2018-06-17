from __future__ import division
import os
import numpy as np
import pandas as pd
from helpers import getattr_else_None
from sklearn.metrics import precision_score, recall_score, f1_score


def calc_metrics(expl):
    """Outputs a dict containing results of the current model (for the current relation).
    Ideally, this function should be called each time after a new model has been fit.

    Arguments:
    - `expl` (Explanator): an instance of the Explanator class
    """
    stats = {}
    stats['model_name'] = expl.model_name

    # relation and data information
    stats['Relation'] = expl.target_relation

    # model parameters
    stats['alpha']             = getattr_else_None(expl.model, 'alpha')
    stats['l1_ratio']          = getattr_else_None(expl.model, 'l1_ratio')
    stats['loss']              = getattr_else_None(expl.model, 'loss')
    stats['penalty']           = getattr_else_None(expl.model, 'penalty')
    stats['max_iter']          = getattr_else_None(expl.model, 'max_iter')
    stats['tol']               = getattr_else_None(expl.model, 'tol')
    stats['class_weight']      = getattr_else_None(expl.model, 'class_weight')
    stats['fit_intercept']     = getattr_else_None(expl.model, 'fit_intercept')
    stats['normalize']         = getattr_else_None(expl.model, 'normalize')
    stats['n_nearby_examples'] = getattr_else_None(expl,       'n_nearby_examples')

    # accuracy
    stats['Test Accuracy']              = expl.model.score(expl.test_x,  expl.test_y)
    stats['True Test Accuracy']         = expl.model.score(expl.test_x,  expl.test_true_y)
    stats['Train Accuracy']             = expl.model.score(expl.train_x, expl.train_y)
    stats['True Train Accuracy']        = expl.model.score(expl.train_x, expl.train_true_y)

    # precision
    stats['Test Precision']             = precision_score(expl.test_y,       expl.predict(expl.test_x))
    stats['True Test Precision']        = precision_score(expl.test_true_y,  expl.predict(expl.test_x))
    stats['Train Precision']            = precision_score(expl.train_y,      expl.predict(expl.train_x))
    stats['True Train Precision']       = precision_score(expl.train_true_y, expl.predict(expl.train_x))

    # recall
    stats['Test Recall']                = recall_score(expl.test_y,       expl.predict(expl.test_x))
    stats['True Test Recall']           = recall_score(expl.test_true_y,  expl.predict(expl.test_x))
    stats['Train Recall']               = recall_score(expl.train_y,      expl.predict(expl.train_x))
    stats['True Train Recall']          = recall_score(expl.train_true_y, expl.predict(expl.train_x))

    # F1 score
    stats['Test F1_score']              = f1_score(expl.test_y,       expl.predict(expl.test_x))
    stats['True Test F1_score']         = f1_score(expl.test_true_y,  expl.predict(expl.test_x))
    stats['Train F1_score']             = f1_score(expl.train_y,      expl.predict(expl.train_x))
    stats['True Train F1_score']        = f1_score(expl.train_true_y, expl.predict(expl.train_x))

    stats['Test Positive Ratio']        = expl.test_y[      expl.test_y==1      ].shape[0]/expl.test_y.shape[0]
    stats['True Test Positive Ratio']   = expl.test_true_y[ expl.test_true_y==1 ].shape[0]/expl.test_true_y.shape[0]
    stats['Train Positive Ratio']       = expl.train_y[     expl.train_y==1     ].shape[0]/expl.train_y.shape[0]
    stats['True Train Positive Ratio']  = expl.train_true_y[expl.train_true_y==1].shape[0]/expl.train_true_y.shape[0]

    stats['Embedding Train Accuracy']   = expl.train_y[expl.train_y == expl.train_true_y].shape[0]/expl.train_y.shape[0]
    stats['Embedding Test Accuracy']    = expl.test_y[ expl.test_y == expl.test_true_y  ].shape[0]/expl.test_y.shape[0]

    # RELEVANT FEATURES
    stats['# Relevant Features'] = expl.explanation[expl.explanation['weight'] != 0].shape[0]
    # NOTE: in the future this should be changed, just because a feature has weight different
    #       than zero it doesn't necessarily mean that it is relevant. We must find a way to
    #       define this "relevance" formally.
    logits_test  = expl.test_x.multiply(expl.model.coef_)
    logits_train = expl.train_x.multiply(expl.model.coef_)
    logits_test.eliminate_zeros()
    logits_train.eliminate_zeros()

    # sum of all non-zero features for each row
    row_sum_feats_test  = expl.test_x.getnnz(axis=-1)
    row_sum_feats_train = expl.train_x.getnnz(axis=-1)
    # sum of all non-zero relevant features for each row
    row_sum_relev_feats_test  = logits_test.getnnz(axis=-1)
    row_sum_relev_feats_train = logits_train.getnnz(axis=-1)

    # sum of all non-zero features
    sum_feats_test  = expl.test_x.getnnz()
    sum_feats_train = expl.train_x.getnnz()
    # sum of all non-zero relevant features
    sum_relev_feats_test  = logits_test.getnnz()
    sum_relev_feats_train = logits_train.getnnz()

    # number of rows that have at least one feature
    n_rows_feats_test  = (row_sum_feats_test > 0).sum()
    n_rows_feats_train = (row_sum_feats_train > 0).sum()
    # number of rows that have at least one relevant feature
    n_rows_relev_feats_test  = (row_sum_relev_feats_test > 0).sum()
    n_rows_relev_feats_train = (row_sum_relev_feats_train > 0).sum()

    n_examples_test  = expl.test_x.shape[0]
    n_examples_train = expl.train_x.shape[0]

    # number of triples for the filtered cases
    # (these metrics are used to calculate the micro average later)
    stats['# Triples Test']  = n_examples_test
    stats['# Triples Train'] = n_examples_train
    stats['# Triples Test (Filter # Features > 0)']  = n_rows_feats_test
    stats['# Triples Train (Filter # Features > 0)'] = n_rows_feats_train
    stats['# Triples Test (Filter # Relevant Features > 0)']  = n_rows_relev_feats_test
    stats['# Triples Train (Filter # Relevant Features > 0)'] = n_rows_relev_feats_train
    stats['# Features Test']  = sum_feats_test
    stats['# Features Train'] = sum_feats_train
    stats['# Relevant Features Test']  = sum_relev_feats_test
    stats['# Relevant Features Train'] = sum_relev_feats_train

    # number of features per example
    stats['Test # Features / Example']  = sum_feats_test  / n_examples_test
    stats['Train # Features / Example'] = sum_feats_train / n_examples_train
    # number of relevant features per example
    stats['Test # Relevant Features / Example']  = sum_relev_feats_test  / n_examples_test
    stats['Train # Relevant Features / Example'] = sum_relev_feats_train / n_examples_train
    # number of features per example (filtered)
    stats['Test # Features / Example (Filter # Features > 0)']  = sum_feats_test  / n_rows_feats_test
    stats['Train # Features / Example (Filter # Features > 0)'] = sum_feats_train / n_rows_feats_train
    # number of relevant features per example (filtered)
    stats['Test # Relevant Features / Example (Filter # Relevant Features > 0)']  = sum_relev_feats_test  / n_rows_relev_feats_test
    stats['Train # Relevant Features / Example (Filter # Relevant Features > 0)'] = sum_relev_feats_train / n_rows_relev_feats_train

    # ratio of triples that have at least one feature
    stats['Test % Examples with # Features > 0']  = n_rows_feats_test  / n_examples_test
    stats['Train % Examples with # Features > 0'] = n_rows_feats_train / n_examples_train
    # ratio of triples that have at least one relevant feature
    stats['Test % Examples with # Relevant Features > 0']  = n_rows_relev_feats_test  / n_examples_test
    stats['Train % Examples with # Relevant Features > 0'] = n_rows_relev_feats_train / n_examples_train

    # accuracy weighted by number of features
    stats['Test Accuracy (Weighted # Features)']           = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=row_sum_feats_test)  if row_sum_feats_test.sum()  > 0 else np.nan
    stats['True Test Accuracy (Weighted # Features)']      = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=row_sum_feats_test)  if row_sum_feats_test.sum()  > 0 else np.nan
    stats['Train Accuracy (Weighted # Features)']          = expl.model.score(expl.train_x, expl.train_y,      sample_weight=row_sum_feats_train) if row_sum_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Weighted # Features)']     = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=row_sum_feats_train) if row_sum_feats_train.sum() > 0 else np.nan
    # accuracy weighted by number of relevant features
    stats['Test Accuracy (Weighted # Relevant Features)']        = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=row_sum_relev_feats_test)  if row_sum_relev_feats_test.sum()  > 0 else np.nan
    stats['True Test Accuracy (Weighted # Relevant Features)']   = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=row_sum_relev_feats_test)  if row_sum_relev_feats_test.sum()  > 0 else np.nan
    stats['Train Accuracy (Weighted # Relevant Features)']       = expl.model.score(expl.train_x, expl.train_y,      sample_weight=row_sum_relev_feats_train) if row_sum_relev_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Weighted # Relevant Features)']  = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=row_sum_relev_feats_train) if row_sum_relev_feats_train.sum() > 0 else np.nan

    # filtered accuracy (only examples with # Features > 0)
    stats['Test Accuracy (Filter # Features > 0)']         = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=(row_sum_feats_test > 0))  if row_sum_feats_test.sum()  > 0 else np.nan
    stats['True Test Accuracy (Filter # Features > 0)']    = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=(row_sum_feats_test > 0))  if row_sum_feats_test.sum()  > 0 else np.nan
    stats['Train Accuracy (Filter # Features > 0)']        = expl.model.score(expl.train_x, expl.train_y,      sample_weight=(row_sum_feats_train > 0)) if row_sum_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Filter # Features > 0)']   = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=(row_sum_feats_train > 0)) if row_sum_feats_train.sum() > 0 else np.nan
    # filtered accuracy (only examples with # Relevant Features > 0)
    stats['Test Accuracy (Filter # Relevant Features > 0)']         = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=(row_sum_relev_feats_test  > 0)) if row_sum_relev_feats_test.sum()  > 0 else np.nan
    stats['True Test Accuracy (Filter # Relevant Features > 0)']    = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=(row_sum_relev_feats_test  > 0)) if row_sum_relev_feats_test.sum()  > 0 else np.nan
    stats['Train Accuracy (Filter # Relevant Features > 0)']        = expl.model.score(expl.train_x, expl.train_y,      sample_weight=(row_sum_relev_feats_train > 0)) if row_sum_relev_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Filter # Relevant Features > 0)']   = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=(row_sum_relev_feats_train > 0)) if row_sum_relev_feats_train.sum() > 0 else np.nan

    # mean rule length
    stats['Test Mean Rule Length']  = (expl.test_x  != 0).multiply(expl.feature_lens).sum() / sum_feats_test  # sum of all rule lengths over the number of examples with at least one feature
    stats['Train Mean Rule Length'] = (expl.train_x != 0).multiply(expl.feature_lens).sum() / sum_feats_train # sum of all rule lengths over the number of examples with at least one feature
    # mean relevant rule length
    stats['Test Mean Relevant Rule Length']  = (logits_test  != 0).multiply(expl.feature_lens).sum() / sum_relev_feats_test  # sum of all rule lengths over the number of examples with at least one feature
    stats['Train Mean Relevant Rule Length'] = (logits_train != 0).multiply(expl.feature_lens).sum() / sum_relev_feats_train # sum of all rule lengths over the number of examples with at least one feature
    return stats


def calc_overall_metrics(dataframe_path, model_info, split):
    """Calculates overall metrics from a per relation metrics dataframe.
    """
    # the dict below has as index the column that should be used as weights when calculating the micro averages.
    # for the macro average, it is not necessary to use weights.
    # we use brackets where the name of the data fold (i.e., Train or Test) should be replaced.
    to_process_metrics = {
        '# Triples {}': [ # metrics that should be weighted by number of examples
            '{} Accuracy',
            'Embedding {} Accuracy',
            '{} F1_score',
            '{} Positive Ratio',
            '{} Precision',
            '{} Recall',
            'True {} Accuracy',
            'True {} F1_score',
            'True {} Positive Ratio',
            'True {} Precision',
            'True {} Recall',
            '{} # Relevant Features / Example',
            '{} # Features / Example',
            '{} # Relevant Features / Example',
            '{} % Examples with # Features > 0',
            '{} % Examples with # Relevant Features > 0',
        ],
        '# Triples {} (Filter # Features > 0)': [ # metrics that should be weighted by number of examples that have at least one feature
            '{} Accuracy (Filter # Features > 0)',
            'True {} Accuracy (Filter # Features > 0)',
            '{} # Features / Example (Filter # Features > 0)',
        ],
        '# Triples {} (Filter # Relevant Features > 0)': [ # metrics that should be weighted by number of examples that have at least one relevat feature
            '{} Accuracy (Filter # Relevant Features > 0)',
            'True {} Accuracy (Filter # Relevant Features > 0)',
            '{} # Relevant Features / Example (Filter # Relevant Features > 0)',
        ],
        '# Features {}': [ # metrics that should be weighted by number of features column
            '{} Accuracy (Weighted # Features)',
            'True {} Accuracy (Weighted # Features)',
            '{} Mean Rule Length',
        ],
        '# Relevant Features {}': [ # metrics that should be weighted by number of relevant features column
            '{} Accuracy (Weighted # Relevant Features)',
            'True {} Accuracy (Weighted # Relevant Features)',
            '{} Mean Relevant Rule Length',
        ],
    }

    metrics_dict = {}
    per_relation_metrics = pd.read_csv(dataframe_path, sep='\t') # dataframe outputted from `calc_metrics`. Each row is a relation.

    # calculate averages ignoring nans (which can happen for metrics that are weighted or filtered).
    for weight_col_name,metrics_list in to_process_metrics.iteritems():
        for metric_name in metrics_list:
            for fold in ['Train', 'Test']:
                m = metric_name.format(fold)
                w = weight_col_name.format(fold)
                metrics_dict[m] = nanaverage(per_relation_metrics[m], weights=per_relation_metrics[w])
                metrics_dict['(macro) ' + m] = np.nanmean(per_relation_metrics[m])

    # additional data
    metrics_dict['dataset_name'] = model_info['dataset_name']
    metrics_dict['model_name'] = model_info['model_name']
    metrics_dict['timestamp'] = model_info['timestamp']
    metrics_dict['split'] = split

    return metrics_dict


def nanaverage(a, weights=None):
    """A wrapper to calculate the weighted average ignoring NANs."""
    indices = ~np.isnan(a)
    return np.average(a[indices], weights=weights[indices])
