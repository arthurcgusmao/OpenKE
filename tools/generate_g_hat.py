import os
import pandas as pd
import numpy as np
import config, models
import itertools
import multiprocessing
import time
from tools import train_test, dataset_tools
from sklearn.neighbors import NearestNeighbors



## KNN (perturbing head, tail and products for each positive example) Generator

def knn_products_generator(k, knn_indices, pos_train_and_valid):
    for idx,row in pos_train_and_valid.iterrows():
        for triple in itertools.product(knn_indices[row['head']][:k], knn_indices[row['tail']][:k], [row['rel']]):
            yield {
                'head': triple[0],
                'tail': triple[1],
                'rel': triple[2]
            }

## General functions for predicting Ĝ

def get_batch_from_generator(triples_iter, batch_size):
    batch_heads = []
    batch_tails = []
    batch_rels = []
    break_ = False

    for i in range(batch_size):
        try:
            triple = next(triples_iter)
        except StopIteration:
            break
        batch_heads.append(triple['head'])
        batch_tails.append(triple['tail'])
        batch_rels.append(triple['rel'])

    return (batch_heads, batch_tails, batch_rels), len(batch_heads)


def filter_positives(heads, tails, rels, preds):
    positive_triples = []
    for idx_n,pred in np.ndenumerate(preds):
        idx = idx_n[0] # ndenumerate works for the dimensional case
        if pred == 1:
            positive_triples.append({
                'head': heads[idx],
                'tail': tails[idx],
                'relation': rels[idx]
            })
    return positive_triples


def predict_g_hat(triples_iterator, batch_size=10000):
    positive_triples = []
    triples_count = 0
    while True:
        (heads, tails, rels), current_batch_size = get_batch_from_generator(triples_iterator, batch_size)
        preds = con.classify(heads, tails, rels, batch_size)
        positive_triples += filter_positives(heads, tails, rels, preds)
        triples_count += current_batch_size
        if current_batch_size < batch_size: # we are at the end of generator
            break
    return positive_triples, triples_count


## Define Pipeline

def pipeline(k, gen, batch_size=100000):
    prediction_info = {}
    prediction_info['knn_time'] = knn_learning_time

    start_time = time.time()
    pos_triples, pred_size = predict_g_hat(
        triples_iterator=gen,
        batch_size=batch_size
    )
    prediction_info['pred_time'] = time.time() - start_time
    prediction_info['positive_size'] = len(pos_triples)
    prediction_info['total_time'] = prediction_info['knn_time'] + prediction_info['pred_time']
    prediction_info['predicted_size'] = pred_size
    prediction_info['k'] = k

    # ensure g_hat dir
    if not os.path.exists(g_hat_path):
        os.makedirs(g_hat_path)

    # save positive triples
    pd.DataFrame(pos_triples).to_csv('{}/positives2id_{}nn.tsv'.format(g_hat_path, k),
                                     sep='\t', index=False, header=False, columns=['head', 'relation', 'tail'])
    return prediction_info



## Main function

def run(import_path, ks):
    """Generate G_hat for different ks.

    Arguments:
    - `import_path` (string): embedding model's directory path
    - `ks` (list): list of numbers of nearest neighbor for which G_hat should be generated.
    """
    if type(ks) != list: raise ValueError("Argument `ks` should be a list.")

    # main variables
    model_info = train_test.read_model_info(import_path)
    dataset_path = './benchmarks/' + model_info['dataset_name']
    g_hat_path = import_path + '/g_hat/'

    # Restore working model
    con = train_test.restore_model(import_path)

    # Read datasets
    print("Reading datasets...")
    train = pd.read_csv(dataset_path + '/train2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'rel'])
    valid = pd.read_csv(dataset_path + '/valid2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'rel'])
    test = pd.read_csv(dataset_path + '/test2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'rel'])
    valid_neg = pd.read_csv(dataset_path + '/valid2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'rel'])
    test_neg = pd.read_csv(dataset_path + '/test2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'rel'])
    pos_train_and_valid = pd.concat([train, valid])
    data = pd.concat([train, valid, test])
    tc_dict = dataset_tools.read_type_constrain_file(dataset_path + '/type_constrain.txt')
    print("OK.")

    # Get all neighbors (Train KNN)
    print("Getting all neighbors (\"training\" KNN)...")
    params = con.get_parameters() # get embedding parameters
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=max(ks), n_jobs=multiprocessing.cpu_count()).fit(params['ent_embeddings'])
    knn_distance, knn_indices = nbrs.kneighbors(params['ent_embeddings'])
    knn_learning_time = time.time() - start_time
    print("OK. KNN learning time: {}".format(knn_learning_time))

    # Predict Ĝ (for all ks)
    prediction_info_list = []
    for k in ks:
        print("Predicting G_hat for k={}...".format(k))
        prediction_info = pipeline(k, knn_products_generator(k, knn_indices, pos_train_and_valid))
        prediction_info_list.append(prediction_info)
        # save prediction info
        pd.DataFrame(prediction_info_list).to_csv(g_hat_path + 'prediction_info.tsv', sep='\t')
        print("G_hat predicted for k={}".format(k))
    print("Run finished.")
