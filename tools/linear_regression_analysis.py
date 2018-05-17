from __future__ import division
import argparse
import itertools
import multiprocessing
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LinearRegression, Lasso
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import time

import config, models
from feature_matrices import parse_matrices_for_relation
import dataset_tools


def get_target_relations(data_set_name):
    if data_set_name == 'NELL':
        data_path = './results/NELL186/TransE/1524632595/pra_explain/results/extract_feat__neg_by_random'
        original_data_path = './benchmarks/NELL186'
        corrupted_data_path = './benchmarks/NELL186/corrupted/train2id_bern_5to1.txt'
    elif data_set_name == 'FB13':
        data_path = './results/FB13/TransE/1524490825/pra_explain/results/extract_feat__neg_by_random'
        original_data_path = './benchmarks/FB13'
        corrupted_data_path = './benchmarks/FB13/corrupted/train2id_bern_2to1.txt'
    elif data_set_name == 'WN11':
        data_path = './results/WN11/TransE/1524623630/pra_explain/results/extract_feat__neg_by_random'
        original_data_path = './benchmarks/WN11'
        corrupted_data_path = './benchmarks/WN11/corrupted/train2id_bern_2to1.txt'
    elif data_set_name == 'g_hat_WN11':
        data_path = './results/WN11/TransE/1524623630/pra_explain/results/results/g_hat_5nn_2negrate_bern'
        original_data_path = './benchmarks/WN11'
        corrupted_data_path = './benchmarks/WN11/corrupted/train2id_bern_2to1.txt'
    elif data_set_name == 'g_hat_NELL':
        data_path = './results/NELL186/TransE/1524632595/pra_explain/results/g_hat_5nn_5negrate_bern'
        original_data_path = './benchmarks/NELL186'
        corrupted_data_path = './benchmarks/NELL186/corrupted/train2id_bern_2to1.txt'
    return data_path, original_data_path, corrupted_data_path, os.listdir(data_path)


def get_reasons(row):
    # Remove zero elements
    reasons = row[row != 0]
    # Select the top n_examples elements
    top_reasons_abs = reasons.abs().nlargest(n=10, keep='first')
    # Create a pandas series with these
    output = pd.Series()
    counter = 1
    for reason, _ in top_reasons_abs.iteritems():
        reason_name, _ = reason.split('=')
        output['reason' + str(counter)] = reason_name
        output['relevance' + str(counter)] = reasons[reason]
        counter = counter + 1
        if counter == 10:
            break
    for i in range(counter, 10):
        output['reason' + str(i)] = "n/a"
        output['relevance' + str(i)] = "n/a"
    return output


def import_embeddings(dataset_name, embedding_model=models.TransE):
    if dataset_name == "NELL186":
        model_timestamp = 1524632595
    elif dataset_name == "FB13":
        model_timestamp = 1524490825
    import_path = './results/{}/{}/{}/'.format(
        dataset_name,
        embedding_model.__name__,
        model_timestamp
    )
    g_hat_path = import_path + '/g_hat/'
    model_info_df = pd.read_csv('{}model_info.tsv'.format(import_path), sep='\t')
    # transform model info into dict with only one "row"
    model_info = model_info_df.to_dict()
    for key, d in model_info.iteritems():
        model_info[key] = d[0]
    # Load the embeddings
    con = config.Config()
    dataset_path = "./benchmarks/{}/".format(model_info['dataset_name'])
    con.set_in_path(dataset_path)
    con.set_test_link_prediction(False)
    con.set_test_triple_classification(True)
    con.set_work_threads(multiprocessing.cpu_count())
    con.set_dimension(int(model_info['k']))
    con.score_norm = model_info['score_norm']
    con.init()
    con.set_model(embedding_model)
    con.import_variables("{}tf_model/model.vec.tf".format(import_path)) # loading model via tensor library

    return con


class Explanator(object):
    def __init__(self, dataset_name, complete_dataframe, target_relation, data_path, original_data_path, corrupted_data_path, model_name="TransE", knn=True):
        self.complete_dataframe = complete_dataframe
        self.target_relation = target_relation
        self.data_path = data_path
        self.original_data_path = original_data_path
        self.corrupted_data_path = corrupted_data_path

        self.valid_exists = True
        self.test_exists = True
        self.stats = {}
        self.stats['Relation'] = target_relation

        # Define the model
        param_grid = [
            {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
             'alpha': [0.01, 0.001, 0.0001]}
        ]

        self.model_definition = SGDClassifier(loss="log",
                                         penalty="elasticnet",
                                         max_iter=100000,
                                         tol=1e-3,
                                         class_weight="balanced",
					                     n_jobs=4)
        self.grid_search = GridSearchCV(self.model_definition, param_grid, n_jobs=4)

        # Get the embedding model
        self.emb_model = import_embeddings(dataset_name, getattr(models, model_name))

        #Define the KNN model
        if knn:
            max_knn_k = 100
            self.embed_params = self.emb_model.get_parameters()
            self.nbrs = NearestNeighbors(n_neighbors=max_knn_k, n_jobs=8).fit(self.embed_params['ent_embeddings'])


    def append_to_dataframe(self):
        self.complete_dataframe = self.complete_dataframe.append(self.stats, ignore_index=True)


    def export_dataframe(self, filepath):
        self.complete_dataframe.to_csv(filepath, mode='a', index=False, header=False)


    def extract_data(self):
        """ Extract data for the target relation from both the original and corrupted datasets """
        # Get original data
        original_data_path = self.original_data_path
        self.entity2id, self.id2entity = dataset_tools.read_name2id_file(os.path.join(original_data_path,'entity2id.txt'))
        self.relation2id, self.id2relation = dataset_tools.read_name2id_file(os.path.join(original_data_path, 'relation2id.txt'))

        true_train = pd.read_csv(self.corrupted_data_path, sep=' ', skiprows=1, names=['e1', 'e2', 'rel', 'true_label'])
        true_valid = pd.read_csv(os.path.join(original_data_path, 'valid.txt'), sep='\t', skiprows=1, names=['head', 'rel_name', 'tail', 'true_label'])
        self.stats['# Triples Valid'] = true_valid[true_valid['rel_name']==target_relation].shape[0]
        true_test = pd.read_csv(os.path.join(original_data_path, 'test.txt'), sep='\t', skiprows=1, names=['head', 'rel_name', 'tail', 'true_label'])
        self.stats['# Triples Test'] = true_test[true_test['rel_name']==target_relation].shape[0]
        # Put all the original data together
        true_data = pd.concat([true_train, true_valid, true_test])

        # Functions to recover entities and relations names
        def apply_id2relation(x):
            return self.id2relation[x]

        def apply_id2entity(x):
            return self.id2entity[x]

        # Add relations and entities names to dataset
        # Training data
        true_train['rel_name'] = true_train['rel'].apply(apply_id2relation)
        self.stats['# Triples Train'] = true_train[true_train['rel_name']==target_relation].shape[0]
        true_train['head'] = true_train['e1'].apply(apply_id2entity)
        true_train['tail'] = true_train['e2'].apply(apply_id2entity)

        # Get target relation data
        # Get the parser for the feature matrices
        parser = parse_matrices_for_relation(data_path, self.target_relation)

        self.train_data = pd.DataFrame(columns=['head', 'tail', 'label'])
        self.train_data['head'] = parser['train_heads']
        self.train_data['tail'] = parser['train_tails']
        self.train_data['label'] = parser['train_labels']

        self.test_data = pd.DataFrame(columns=['head', 'tail', 'label'])
        self.test_data['head'] = parser['test_heads']
        self.test_data['tail'] = parser['test_tails']
        self.test_data['label'] = parser['test_labels']

        self.valid_data = pd.DataFrame(columns=['head', 'tail', 'label'])
        self.valid_data['head'] = parser['valid_heads']
        self.valid_data['tail'] = parser['valid_tails']
        self.valid_data['label'] = parser['valid_labels']

        # Get true labels for target relations (training data)
        rel_true_train = true_train[true_train['rel_name']==self.target_relation].copy()
        self.train_data = self.train_data.merge(rel_true_train[['head', 'tail', 'true_label']].drop_duplicates(subset=['head', 'tail']), how='left', on=['head', 'tail'])
        self.train_data = self.train_data.fillna(-1)
        # separate x (features) and y (labels) for training data
        self.train_y = self.train_data.pop('label')
        self.true_train_y = self.train_data.pop('true_label')
        self.train_x = parser['train_X']

        # Save features names and number of features before pre-processing
        self.columns = parser['vectorizer'].get_feature_names()
        self.stats['# Features'] = len(self.columns)

        # Get true labels for target relations (validation data)
        rel_true_valid = true_valid[true_valid['rel_name']==self.target_relation].copy()

        if rel_true_valid.empty or self.valid_data.empty:
            self.valid_exists = False
        else:
            self.valid_data = self.valid_data.merge(rel_true_valid[['head', 'tail', 'true_label']].drop_duplicates(subset=['head', 'tail']), how='left', on=['head', 'tail'])
            self.valid_data = self.valid_data.fillna(-1)
            # Validation data
            self.valid_y = self.valid_data.pop('label')
            self.true_valid_y = self.valid_data.pop('true_label')
            self.valid_x = parser['valid_X']
        # Get true labels for target relations (test data)
        rel_true_test = true_test[true_test['rel_name']==self.target_relation].copy()
        if rel_true_test.empty or self.test_data.empty:
            self.test_exists = False
        else:
            self.test_data = self.test_data.merge(rel_true_test[['head', 'tail', 'true_label']].drop_duplicates(subset=['head', 'tail']), how='left', on=['head', 'tail'])
            self.test_data = self.test_data.fillna(-1)
            # Test data
            self.test_y = self.test_data.pop('label')
            self.true_test_y = self.test_data.pop('true_label')
            self.test_x = parser['test_X']

        return True

    def train_local_logit(self, head, tail):
        """ Train and evaluate the model locally """
        # Get the nearest neighbors
        _, head_indices = self.nbrs.kneighbors(self.embed_params['ent_embeddings'][self.entity2id[head]].reshape(1, -1))
        _, tail_indices = self.nbrs.kneighbors(self.embed_params['ent_embeddings'][self.entity2id[tail]].reshape(1, -1))
        # Get all the corresponding training examples
        examples_indices = []
        for head_index in head_indices[0][1:]:
            examples_indices.extend(self.train_data.index[self.train_data['head'] == self.id2entity[head_index]].tolist())
        for tail_index in tail_indices[0][1:]:
            examples_indices.extend(self.train_data.index[self.train_data['tail'] == self.id2entity[tail_index]].tolist())
        examples_indices = sorted(examples_indices)
        # Train a logit on those examples
        print "Training with ", len(examples_indices)
        x = self.train_x[examples_indices, :]
        y = self.train_y.iloc[examples_indices]
        self.model_definition.fit(x, y)
        # Get the features of the test example
        test_index = self.test_data.index[(self.test_data['head'] == head) & (self.test_data['tail'] == tail)]
        print "INDEX ", test_index
        test_x = self.test_x[test_index, :]
        test_y = self.test_y.iloc[test_index]
        prediction = self.model_definition.predict_proba(test_x)[:, 1]
        print "The triple has been predicted as ", prediction, " when should have been ", test_y

    def train_local_regression(self, head, tail):
        """ Train and evaluate the model locally """
        # Get the nearest neighbors
        _, head_indices = self.nbrs.kneighbors(self.embed_params['ent_embeddings'][self.entity2id[head]].reshape(1, -1))
        _, tail_indices = self.nbrs.kneighbors(self.embed_params['ent_embeddings'][self.entity2id[tail]].reshape(1, -1))
        # Get all the corresponding training examples
        examples_indices = []
        for head_index in head_indices[0][1:]:
            examples_indices.extend(self.train_data.index[self.train_data['head'] == self.id2entity[head_index]].tolist())
        for tail_index in tail_indices[0][1:]:
            examples_indices.extend(self.train_data.index[self.train_data['tail'] == self.id2entity[tail_index]].tolist())

        # Train a logit on those examples
        print "Training with ", len(examples_indices)
        x = self.train_x[examples_indices, :]
        x_info = self.train_data.iloc[examples_indices]

        def get_embed_y(row):
            # Get head, tail and rel IDs
            head = [self.entity2id[row['head']]]
            tail = [self.entity2id[row['tail']]]
            rel = [self.relation2id[self.target_relation]]
            predict = self.emb_model.test_step(head, tail, rel)
            return predict[0]

        x_info['EmbPrediction'] = x_info.apply(get_embed_y, axis=1)
        y = x_info.pop('EmbPrediction')

        # Define the model
        param_grid = [
            {'alpha': [0.1, 0.3, 0.9, 1.0]}
        ]

        model_definition = Lasso(max_iter=10000,
                                 fit_intercept=True)
        grid_search = GridSearchCV(model_definition, param_grid, n_jobs=4)

        grid_search.fit(x, y)
        # Get the features of the test example
        test_index = self.test_data.index[(self.test_data['head'] == head) & (self.test_data['tail'] == tail)]
        test_x = self.test_x[test_index, :]
        self.test_data['EmbPrediction'] = self.test_data.apply(get_embed_y, axis=1)
        test_y = self.test_data.pop('EmbPrediction')
        prediction = grid_search.predict(test_x)
        print "The triple has been predicted as ", prediction, " when should have been ", test_y.iloc[test_index].values

    def train(self):
        """ Train and evaluate the model """

        # Search for the best parameters
    	try:
            training_examples = np.concatenate((self.train_x, self.valid_x), axis=0)
            training_examples_y = np.concatenate((self.train_y, self.valid_y), axis=0)
    	except:
            training_examples = self.train_x
            training_examples_y = self.train_y

        try:
            self.grid_search.fit(training_examples, training_examples_y)
            #  self.grid_search.fit(self.train_x, self.train_y)
        except ValueError:
            print("Not possible to fit a logit for this relation because it contains a single class.")

        alpha = self.grid_search.best_params_['alpha']
        self.stats['alpha'] = alpha
        l1_ratio = self.grid_search.best_params_['l1_ratio']
        self.stats['l1_ratio'] = l1_ratio
        # Fit the best model
        # We need to refit it because GridSearchCV does give access to coef_
        self.model = SGDClassifier(l1_ratio=l1_ratio, alpha=alpha, loss="log", penalty="elasticnet",
                      max_iter=100000, tol=1e-3, class_weight="balanced")

        self.model.fit(self.train_x, self.train_y)

        ### Get evaluation metrics
        # Get accuracy
        if self.test_exists:
            self.stats['Test Accuracy'] = self.model.score(self.test_x, self.test_y)
            self.stats['True Test Accuracy'] = self.model.score(self.test_x, self.true_test_y)
        else:
            self.stats['Test Accuracy'] = -1
            self.stats['True Test Accuracy'] = -1
        if self.valid_exists:
            self.stats['Valid Accuracy'] = self.model.score(self.valid_x, self.valid_y)
            self.stats['True Valid Accuracy'] = self.model.score(self.valid_x, self.true_valid_y)
        else:
            self.stats['Valid Accuracy'] = -1
            self.stats['True Valid Accuracy'] = -1
        self.stats['Train Accuracy'] = self.model.score(self.train_x, self.train_y)
        self.stats['True Train Accuracy'] = self.model.score(self.train_x, self.true_train_y)

        # Get precision
        if self.test_exists:
            self.stats['Test Precision'] = precision_score(self.test_y, self.model.predict(self.test_x))
            self.stats['True Test Precision'] = precision_score(self.true_test_y, self.model.predict(self.test_x))
        else:
            self.stats['Test Precision'] = -1
            self.stats['True Test Precision'] = -1
        if False: #self.valid_exists:
            self.stats['Valid Precision'] = precision_score(self.valid_y, self.model.predict(self.valid_x))
            self.stats['True Valid Precision'] = precision_score(self.true_valid_y, self.model.predict(self.valid_x))
        else:
            self.stats['Valid Precision'] = -1
            self.stats['True Valid Precision'] = -1
        self.stats['Train Precision'] = precision_score(self.train_y, self.model.predict(self.train_x))
        self.stats['True Train Precision'] = precision_score(self.true_train_y, self.model.predict(self.train_x))

        # Get recall
        if self.test_exists:
            self.stats['Test Recall'] = recall_score(self.test_y, self.model.predict(self.test_x))
            self.stats['True Test Recall'] = recall_score(self.true_test_y, self.model.predict(self.test_x))
        else:
            self.stats['Test Recall'] = -1
            self.stats['True Test Recall'] = -1
        if self.valid_exists:
            self.stats['Valid Recall'] = recall_score(self.valid_y, self.model.predict(self.valid_x))
            self.stats['True Valid Recall'] = recall_score(self.true_valid_y, self.model.predict(self.valid_x))
        else:
            self.stats['Valid Recall'] = -1
            self.stats['True Valid Recall'] = -1
        self.stats['Train Recall'] = recall_score(self.train_y, self.model.predict(self.train_x))
        self.stats['True Train Recall'] = recall_score(self.true_train_y, self.model.predict(self.train_x))

        # Get F1 score
        if self.test_exists:
            self.stats['Test F1_score'] = f1_score(self.test_y, self.model.predict(self.test_x))
            self.stats['True Test F1_score'] = f1_score(self.true_test_y, self.model.predict(self.test_x))
        else:
            self.stats['Test F1_score'] = -1
            self.stats['True Test F1_score'] = -1
        if self.valid_exists:
            self.stats['Valid F1_score'] = f1_score(self.valid_y, self.model.predict(self.valid_x))
            self.stats['True Valid F1_score'] = f1_score(self.true_valid_y, self.model.predict(self.valid_x))
        else:
            self.stats['Valid F1_score'] = -1
            self.stats['True Valid F1_score'] = -1
        self.stats['Train F1_score'] = f1_score(self.train_y, self.model.predict(self.train_x))
        self.stats['True Train F1_score'] = f1_score(self.true_train_y, self.model.predict(self.train_x))

        if self.test_exists:
            self.stats['Test Positive Ratio'] = self.test_y[self.test_y==1].shape[0]/self.test_y.shape[0]
            self.stats['True Test Positive Ratio'] = self.true_test_y[self.true_test_y==1].shape[0]/self.true_test_y.shape[0]
        else:
            self.stats['Test Positive Ratio'] = -1
            self.stats['True Test Positive Ratio'] = -1
        if self.valid_exists:
            self.stats['Valid Positive Ratio'] = self.valid_y[self.valid_y==1].shape[0]/self.valid_y.shape[0]
            self.stats['True Valid Positive Ratio'] = self.true_valid_y[self.true_valid_y==1].shape[0]/self.true_valid_y.shape[0]
        else:
            self.stats['Valid Positive Ratio'] = -1
            self.stats['True Valid Positive Ratio'] = -1

        self.stats['Train Positive Ratio'] = self.train_y[self.train_y==1].shape[0]/self.train_y.shape[0]
        self.stats['True Train Positive Ratio'] = self.true_train_y[self.true_train_y==1].shape[0]/self.true_train_y.shape[0]

        self.stats['Train Embedding Accuracy'] = self.train_y[self.train_y == self.true_train_y].shape[0]/self.train_y.shape[0]
        if self.test_exists:
            self.stats['Test Embedding Accuracy'] = self.test_y[self.test_y == self.true_test_y].shape[0]/self.test_y.shape[0]
        else:
            self.stats['Test Embedding Accuracy'] = -1
        if self.valid_exists:
            self.stats['Valid Embedding Accuracy'] = self.valid_y[self.valid_y == self.true_valid_y].shape[0]/self.valid_y.shape[0]
        else:
            self.stats['Valid Embedding Accuracy'] = -1

    def explain_per_example(self, data_path, data_type, n_examples=10):
        coefficients = self.model.coef_.reshape(-1,1)
        if data_type == 'train':
            x = self.train_x
            y_hat = self.train_y
            y = self.true_train_y
            data = self.train_data
        elif self.test_exists:
            x = self.test_x
            y_hat = self.test_y
            y = self.true_test_y
            data = self.test_data
        else:
            return ''
        # Define the maximum number of examples
        n_examples = min(n_examples, x.shape[0])
        # Select n_examples samples
        index = np.random.choice(x.shape[0], n_examples, replace=False)
        features = x[index, :]
        features = features.todense()
        # Compute those that are relevant
        repeated_coefficients = np.repeat(coefficients.T, n_examples, axis=0)
        explanations = np.multiply(features, repeated_coefficients)
        # Define a pandas DataFrame to hold the information
        examples_df = pd.DataFrame(explanations, columns=self.columns)
        examples_df.columns = self.columns

        final_reasons = examples_df.apply(get_reasons, axis=1)
        final_reasons['head'] = data.iloc[index]['head'].values
        final_reasons['tail'] = data.iloc[index]['tail'].values
        answers = self.model.predict_proba(features)[:, 1]
        final_reasons['y_logit'] = answers
        final_reasons['y_hat'] = y_hat
        final_reasons['y'] = y
        final_reasons.to_csv(data_path  + '/' + self.target_relation + '/' + self.target_relation + '.csv', index=False)
        return final_reasons

    def explain(self):
        """ Explain the model using the coefficients """
        # Extract the coefficients
        self.coefficients = self.model.coef_.reshape(-1,1)

        self.explanation = pd.DataFrame(self.coefficients, columns=['scores'])
        self.explanation['path'] = self.columns
        self.explanation = self.explanation.sort_values(by="scores", ascending=False)
        explanation = self.explanation[self.explanation['scores'] != 0]
        self.most_relevant_variables = pd.concat([explanation.iloc[0:10], explanation.iloc[-10:-1]])
        self.stats['# Relevant Features'] = self.explanation[self.explanation['scores'] != 0].shape[0]

    def report(self):
        file_path = os.path.join(self.data_path, self.target_relation, self.target_relation + '_explained.txt')
        open(file_path, 'w+').close()
        np.savetxt(file_path, self.most_relevant_variables.values, fmt= '%s')
        with open(file_path, 'a') as f:
            f.write("\n---------------------------------------------")
            f.write("\nNumber of relevant variables : %0.2f" % self.stats['# Relevant Features'])
            f.write("\nTotal number of variables : %0.2f" % self.stats['# Features'])
            f.write("\n\nDataset Positive Ratio:")
            f.write("\n   Test : %0.2f" % self.stats['Test Positive Ratio'])
            f.write("\n   True test: %0.2f" % self.stats['True Test Positive Ratio'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid Positive Ratio'])
            f.write("\n   True Valid: %0.2f" % self.stats['True Valid Positive Ratio'])
            f.write("\n   Train: %0.2f" % self.stats['Train Positive Ratio'])
            f.write("\n   True Train: %0.2f" % self.stats['True Train Positive Ratio'])
            f.write("\n---------------------------------------------")
            f.write("\n\nEmbedding Accuracy:")
            f.write("\n   Test: %0.2f" % self.stats['Test Embedding Accuracy'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid Embedding Accuracy'])
            f.write("\n   Train: %0.2f" % self.stats['Train Embedding Accuracy'])
            f.write("\n\nAccuracy:")
            f.write("\n   Test: %0.2f" % self.stats['Test Accuracy'])
            f.write("\n   True test: %0.2f" % self.stats['True Test Accuracy'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid Accuracy'])
            f.write("\n   True Valid: %0.2f" % self.stats['True Valid Accuracy'])
            f.write("\n   Train: %0.2f" % self.stats['Train Accuracy'])
            f.write("\n   True Train: %0.2f" % self.stats['True Train Accuracy'])

            f.write("\n\nPrecision:")
            f.write("\n   Test: %0.2f" % self.stats['Test Precision'])
            f.write("\n   True test: %0.2f" % self.stats['True Test Precision'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid Precision'])
            f.write("\n   True Valid: %0.2f" % self.stats['True Valid Precision'])
            f.write("\n   Train: %0.2f" % self.stats['Train Precision'])
            f.write("\n   True Train: %0.2f" % self.stats['True Train Precision'])

            f.write("\n\nRecall: ")
            f.write("\n   Test: %0.2f" % self.stats['Test Recall'])
            f.write("\n   True test: %0.2f" % self.stats['True Test Recall'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid Recall'])
            f.write("\n   True Valid: %0.2f" % self.stats['True Valid Recall'])
            f.write("\n   Train: %0.2f" % self.stats['Train Recall'])
            f.write("\n   True Train: %0.2f" % self.stats['True Train Recall'])

            f.write("\n\nF1_score:")
            f.write("\n   Test: %0.2f" % self.stats['Test F1_score'])
            f.write("\n   True test: %0.2f" % self.stats['True Test F1_score'])
            f.write("\n   Valid: %0.2f" % self.stats['Valid F1_score'])
            f.write("\n   True Valid: %0.2f" % self.stats['True Valid F1_score'])
            f.write("\n   Train: %0.2f" % self.stats['Train F1_score'])
            f.write("\n   True Train: %0.2f" % self.stats['True Train F1_score'])
            f.write("\n---------------------------------------------")
            f.write("\n" + str(self.model.get_params()))

if __name__ == '__main__':
    # parser = argparse.Argumentparser()

    # parser.add_argument(
    #     '--output', '-o',
    #     type=str,
    #     default='stats',
    #     help=''''stats' for relation statistics,
    #     'per-example' for per-example explanations '''
    # )

    # args = parser.parse_args()

    columns = ['Relation', '# Triples Train', '# Triples Valid', '# Triples Test', '# Features', '# Relevant Features',
               'Test Embedding Accuracy', 'Valid Embedding Accuracy', 'Train Embedding Accuracy',
               'Test Positive Ratio', 'True Test Positive Ratio', 'Valid Positive Ratio', 'True Valid Positive Ratio', 'Train Positive Ratio', 'True Train Positive Ratio',
               'Test Accuracy', 'True Test Accuracy', 'Valid Accuracy', 'True Valid Accuracy', 'Train Accuracy', 'True Train Accuracy',
               'Test Precision', 'True Test Precision', 'Valid Precision', 'True Valid Precision', 'Train Precision', 'True Train Precision',
               'Test Recall', 'True Test Recall', 'Valid Recall', 'True Valid Recall', 'Train Recall', 'True Train Recall',
               'Test F1_score', 'True Test F1_score', 'Valid F1_score', 'True Valid F1_score', 'Train F1_score', 'True Train F1_score',
               'l1_ratio', 'alpha'
              ]

    data_base_names = ['FB13']
    for data_base_name in data_base_names:

        data_path, original_data_path, corrupted_data_path, target_relations = get_target_relations(data_base_name)

	# Export dataframe headers to csv
        complete_dataframe = pd.DataFrame(columns=columns)
        complete_dataframe.to_csv(data_path + data_base_name + '.csv', index=False)
        target_relations = ['nationality']
        for target_relation in target_relations:
            print("Training on " + target_relation + " relations")
            exp = Explanator(data_base_name, complete_dataframe, target_relation, data_path, original_data_path, corrupted_data_path)
            if exp.extract_data():
                exp.train_local_regression("john_forsyth", "roman_empire")
                # exp.train():
                # print('    Generating explanation')
                # exp.explain()
                # print('    Generating report')
                # exp.report()
                # print('    Saving to csv')
                # exp.append_to_dataframe()
                # exp.export_dataframe(data_path + data_base_name + '.csv')
                # print('    Generating per-example explanations')
                # exp.explain_per_example(data_path, 'test')
            else:
                print("No test data for ", target_relation, " data")
