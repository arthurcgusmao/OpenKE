from __future__ import division
import os
import numpy as np
import pandas as pd



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
    stats['# Triples Train'] = len(expl.train_heads)
    stats['# Triples Test '] = len(expl.test_heads )
    # stats['# Triples Valid'] = ??? # we are now using the CV's validation sets

    # model parameters
    stats['alpha']             = getattr_else_None(expl.model, 'alpha'            )
    stats['l1_ratio']          = getattr_else_None(expl.model, 'l1_ratio'         )
    stats['loss']              = getattr_else_None(expl.model, 'loss'             )
    stats['penalty']           = getattr_else_None(expl.model, 'penalty'          )
    stats['max_iter']          = getattr_else_None(expl.model, 'max_iter'         )
    stats['tol']               = getattr_else_None(expl.model, 'tol'              )
    stats['class_weight']      = getattr_else_None(expl.model, 'class_weight'     )
    stats['fit_intercept']     = getattr_else_None(expl.model, 'fit_intercept'    )
    stats['normalize']         = getattr_else_None(expl.model, 'normalize'        )
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

    stats['Train Embedding Accuracy']   = expl.train_y[expl.train_y == expl.train_true_y].shape[0]/expl.train_y.shape[0]
    stats['Test Embedding Accuracy']    = expl.test_y[ expl.test_y == expl.test_true_y  ].shape[0]/expl.test_y.shape[0]

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
    n_examples_total = n_examples_train + n_examples_test

    # number of features per example
    stats['Test # Features / Example'] = sum_feats_test / n_examples_test
    stats['Train # Features / Example'] = sum_feats_train / n_examples_train
    stats['Total # Features / Example'] = (sum_feats_train + sum_feats_test) / n_examples_total
    # number of relevant features per example
    stats['Test # Relevant Features / Example'] = sum_relev_feats_test / n_examples_test
    stats['Train # Relevant Features / Example'] = sum_relev_feats_train / n_examples_train
    stats['Total # Relevant Features / Example'] = (sum_relev_feats_train + sum_relev_feats_test) / n_examples_total

    # number of triples for the filtered cases
    # (these metrics are used to calculate the micro average later)
    stats['# Triples Test  (Filter # Features > 0)'] = n_rows_feats_test
    stats['# Triples Train (Filter # Features > 0)'] = n_rows_feats_train
    stats['# Triples Test  (Filter # Relevant Features > 0)'] = n_rows_relev_feats_test
    stats['# Triples Train (Filter # Relevant Features > 0)'] = n_rows_relev_feats_train

    # ratio of triples that have at least one feature
    stats['Test % Examples with # Features > 0'] = n_rows_feats_test / n_examples_test
    stats['Train % Examples with # Features > 0'] = n_rows_feats_train / n_examples_train
    stats['Total % Examples with # Features > 0'] = (n_rows_feats_train + n_rows_feats_test) / n_examples_total
    # ratio of triples that have at least one relevant feature
    stats['Test % Examples with # Relevant Features > 0'] = n_rows_relev_feats_test / n_examples_test
    stats['Train % Examples with # Relevant Features > 0'] = n_rows_relev_feats_train / n_examples_train
    stats['Total % Examples with # Relevant Features > 0'] = (n_rows_relev_feats_train + n_rows_relev_feats_test) / n_examples_total

    # accuracy weighted by number of features
    stats['Test Accuracy (Weighted # Features)']           = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=row_sum_feats_test) if row_sum_feats_test.sum() > 0 else np.nan
    stats['True Test Accuracy (Weighted # Features)']      = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=row_sum_feats_test) if row_sum_feats_test.sum() > 0 else np.nan
    stats['Train Accuracy (Weighted # Features)']          = expl.model.score(expl.train_x, expl.train_y,      sample_weight=row_sum_feats_train) if row_sum_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Weighted # Features)']     = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=row_sum_feats_train) if row_sum_feats_train.sum() > 0 else np.nan
    # accuracy weighted by number of relevant features
    stats['Test Accuracy (Weighted # Relevant Features)']           = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=row_sum_relev_feats_test) if row_sum_relev_feats_test.sum() > 0 else np.nan
    stats['True Test Accuracy (Weighted # Relevant Features)']      = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=row_sum_relev_feats_test) if row_sum_relev_feats_test.sum() > 0 else np.nan
    stats['Train Accuracy (Weighted # Relevant Features)']          = expl.model.score(expl.train_x, expl.train_y,      sample_weight=row_sum_relev_feats_train) if row_sum_relev_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Weighted # Relevant Features)']     = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=row_sum_relev_feats_train) if row_sum_relev_feats_train.sum() > 0 else np.nan

    # filtered accuracy (only examples with # Features > 0)
    stats['Test Accuracy (Filter # Features > 0)']         = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=(row_sum_feats_test > 0)) if row_sum_feats_test.sum() > 0 else np.nan
    stats['True Test Accuracy (Filter # Features > 0)']    = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=(row_sum_feats_test > 0)) if row_sum_feats_test.sum() > 0 else np.nan
    stats['Train Accuracy (Filter # Features > 0)']        = expl.model.score(expl.train_x, expl.train_y,      sample_weight=(row_sum_feats_train > 0)) if row_sum_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Filter # Features > 0)']   = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=(row_sum_feats_train > 0)) if row_sum_feats_train.sum() > 0 else np.nan
    # filtered accuracy (only examples with # Relevant Features > 0)
    stats['Test Accuracy (Filter # Relevant Features > 0)']         = expl.model.score(expl.test_x,  expl.test_y,       sample_weight=(row_sum_relev_feats_test > 0)) if row_sum_relev_feats_test.sum() > 0 else np.nan
    stats['True Test Accuracy (Filter # Relevant Features > 0)']    = expl.model.score(expl.test_x,  expl.test_true_y,  sample_weight=(row_sum_relev_feats_test > 0)) if row_sum_relev_feats_test.sum() > 0 else np.nan
    stats['Train Accuracy (Filter # Relevant Features > 0)']        = expl.model.score(expl.train_x, expl.train_y,      sample_weight=(row_sum_relev_feats_train > 0)) if row_sum_relev_feats_train.sum() > 0 else np.nan
    stats['True Train Accuracy (Filter # Relevant Features > 0)']   = expl.model.score(expl.train_x, expl.train_true_y, sample_weight=(row_sum_relev_feats_train > 0)) if row_sum_relev_feats_train.sum() > 0 else np.nan

    # mean rule length
    stats['Test Mean Rule Length (Filter # Relevant Features > 0)']  = (logits_test != 0).multiply(expl.feature_lens).sum() / sum_relev_feats_test # sum of all rule lengths over the number of examples with at least one feature
    stats['Train Mean Rule Length (Filter # Relevant Features > 0)'] = (logits_train != 0).multiply(expl.feature_lens).sum() / sum_relev_feats_train # sum of all rule lengths over the number of examples with at least one feature
    return stats




def calc_overall_metrics(dataframe_path, model_info, split):
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
