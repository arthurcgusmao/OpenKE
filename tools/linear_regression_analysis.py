from __future__ import division
import argparse
import itertools
import multiprocessing
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import time

import config, models
# from tools.feature_matrices import parse_matrices_for_relation
from tools.feature_matrices import parse_feature_matrix
from tools import dataset_tools, train_test


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



class Explanator(object):
    def __init__(self, emb_model_path, ground_truth_dataset_path, target_relation, data_path):
        self.target_relation = target_relation
        self.data_path = data_path
        self.ground_truth_dataset_path = ground_truth_dataset_path

        # Define the model
        param_grid = [{
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'alpha': [0.01, 0.001, 0.0001],
            'loss': "log",
            'penalty': "elasticnet",
            'max_iter': 100000,
            'tol': 1e-3,
            'class_weight': "balanced",
            'n_jobs': 4,
        }]

        self.model_definition = SGDClassifier()
        self.grid_search = GridSearchCV(self.model_definition, param_grid, n_jobs=4)

        # Get the embedding model
        self.emb_model = train_test.restore_model(emb_model_path)



    def define_knn():
        max_knn_k = 100
        self.embed_params = self.emb_model.get_parameters()
        self.nbrs = NearestNeighbors(n_neighbors=max_knn_k, n_jobs=4).fit(self.embed_params['ent_embeddings'])



    def read_data(self, pra_results_dpath, target_relation):
        """Read embedding predicted data for the target relation from the split (i.e., from data
        whose features were extracted and whose labels are predictions from the embedding model).

        Data is stored into the following lists or numpy ndarrays:

        - self.train_heads: contains the head for each training example
        - self.train_tails: contains the tail for each training example
        - self.train_y: contains the label (by the embedding model) for each training example
        - self.train_x: contains the extracted features (by PRA/SFE) for each training example

        - self.test_heads: contains the head for each testing example
        - self.test_tails: contains the tail for each testing example
        - self.test_y: contains the label (by the embedding model) for each testing example
        - self.test_x: contains the extracted features (by PRA/SFE) for each testing example

        If there is no test data, then we do not even train an explanation to begin with, and
        `False` is returned. If everything went alright, then `True` is returned.
        """
        ###################################################
        ###################################################
        ###################################################
        train_fpath = "{}/{}/train.tsv".format(pra_results_dpath, target_relation)
        valid_fpath = "{}/{}/valid.tsv".format(pra_results_dpath, target_relation)
        test_fpath  = "{}/{}/test.tsv" .format(pra_results_dpath, target_relation)

        # check if `test.tsv` and `train.tsv` are present
        if not os.path.exists(test_fpath):
            print("There is no test file for relation `{}`, skipping.".format(target_relation))
            return False
        else:
            if not os.path.exists(train_fpath): raise IOError('`train.tsv` not present for relation `{}`'.format(target_relation))

        # read train data (always present)
        self.train_heads, self.train_tails, self.train_y, train_feat_dicts = parse_feature_matrix(train_fpath)
        v = DictVectorizer(sparse=True)
        v.fit(train_feat_dicts)
        self.train_x = v.transform(train_feat_dicts)
        self.feature_names = v.get_feature_names()

        # read valid data (may not be present)
        if os.path.exists(valid_fpath):
            valid_heads, valid_tails, valid_y, valid_feat_dicts = parse_feature_matrix(valid_fpath)
            valid_x = v.transform(valid_feat_dicts)
            # we merge validation with training data, because the GridSearchCV creates the valid split automatically
            self.train_heads += valid_heads
            self.train_tails += valid_tails
            self.train_x = np.concatenate((self.train_x, self.valid_x), axis=0)
            self.train_y = np.concatenate((self.train_y, self.valid_y), axis=0)

        # read test data (always present)
        self.test_heads, self.test_tails, self.test_y, test_feat_dicts = parse_feature_matrix(test_fpath)
        self.test_x = v.transform(test_feat_dicts)

        # check that there are not only positive or negative labels in training set
        if len(np.unique(self.train_y)) <= 1:
            print("Not possible to train explainable model in relation `{}` because training set contains a single class.".format(target_relation))
            return False

        return True



    def read_ground_truth_labels(self, dataset_path, target_relation):
        """Read ground truth data from the original dataset and extract labels for the data
        present in the split (i.e., data whose features are extracted and that is present in
        `self.train_x`, etc.).

        We get the original data from a list of positive triples in order to dispense with the need
        for having the corrupted data path (that may change from model to model).
        """
        entity2id, id2entity = dataset_tools.read_name2id_file(os.path.join(dataset_path,'entity2id.txt'))
        relation2id, id2relation = dataset_tools.read_name2id_file(os.path.join(dataset_path, 'relation2id.txt'))

        # these files have no labels, they are all positive instances
        gt_train2id = pd.read_csv(os.path.join(dataset_path, 'train2id.txt'), skiprows=1, sep=' ', columns=['head', 'tail', 'relation'])
        gt_valid2id = pd.read_csv(os.path.join(dataset_path, 'valid2id.txt'), skiprows=1, sep=' ', columns=['head', 'tail', 'relation'])
        gt_test2id  = pd.read_csv(os.path.join(dataset_path, 'test2id.txt' ), skiprows=1, sep=' ', columns=['head', 'tail', 'relation'])

        # merge train and validation data
        gt_train2id = pd.concat((gt_train2id, gt_valid2id))

        # get id of target relation
        target_relation_id = relation2id[target_relation]

        # filter data to get only triples whose relation is the target relation
        gt_train2id_filt = gt_train2id.loc[gt_train2id['relation'] == target_relation_id]
        gt_test2id_filt  =  gt_test2id.loc[ gt_test2id['relation'] == target_relation_id]

        # compare split data with ground truth data and create labels
        heads2id = [entity2id[h] for h in self.train_heads]
        tails2id = [entity2id[t] for t in self.train_tails]

        self.train_true_y = []
        for head,tail in zip(heads2id, tails2id):
            matches = len(gt_train2id_filt.loc[
                (gt_train2id_filt['head'] == head) &
                (gt_train2id_filt['tail'] == tail)
            ])
            self.train_true_y.append(1 if matches > 0 else -1)

        self.test_true_y = []
        for head,tail in zip(heads2id, tails2id):
            matches = len(gt_test2id_filt.loc[
                (gt_test2id_filt['head'] == head) &
                (gt_test2id_filt['tail'] == tail)
            ])
            self.test_true_y.append(1 if matches > 0 else -1)


    def get_stats():
        stats = {}

        # relation and data information
        stats['Relation'] = self.target_relation
        stats['# Triples Train'] = len(train_heads)
        stats['# Triples Test '] = len(test_heads )
        # stats['# Triples Valid'] = ??? # we are now using the CV's validation sets

        # model parameters
        stats['alpha']    = self.grid_search.best_params_['alpha'   ]
        stats['l1_ratio'] = self.grid_search.best_params_['l1_ratio']

        # accuracy
        stats['Test Accuracy']              = self.model.score(self.test_x,  self.test_y)
        stats['True Test Accuracy']         = self.model.score(self.test_x,  self.test_true_y)
        stats['Train Accuracy']             = self.model.score(self.train_x, self.train_y)
        stats['True Train Accuracy']        = self.model.score(self.train_x, self.train_true_y)

        # precision
        stats['Test Precision']             = precision_score(self.test_y,       self.model.predict(self.test_x))
        stats['True Test Precision']        = precision_score(self.test_true_y,  self.model.predict(self.test_x))
        stats['Train Precision']            = precision_score(self.train_y,      self.model.predict(self.train_x))
        stats['True Train Precision']       = precision_score(self.train_true_y, self.model.predict(self.train_x))

        # recall
        stats['Test Recall']                = recall_score(self.test_y,       self.model.predict(self.test_x))
        stats['True Test Recall']           = recall_score(self.test_true_y,  self.model.predict(self.test_x))
        stats['Train Recall']               = recall_score(self.train_y,      self.model.predict(self.train_x))
        stats['True Train Recall']          = recall_score(self.train_true_y, self.model.predict(self.train_x))

        # F1 score
        stats['Test F1_score']              = f1_score(self.test_y,       self.model.predict(self.test_x))
        stats['True Test F1_score']         = f1_score(self.test_true_y,  self.model.predict(self.test_x))
        stats['Train F1_score']             = f1_score(self.train_y,      self.model.predict(self.train_x))
        stats['True Train F1_score']        = f1_score(self.train_true_y, self.model.predict(self.train_x))

        stats['Test Positive Ratio']        = self.test_y[      self.test_y==1      ].shape[0]/self.test_y.shape[0]
        stats['True Test Positive Ratio']   = self.test_true_y[ self.test_true_y==1 ].shape[0]/self.test_true_y.shape[0]
        stats['Train Positive Ratio']       = self.train_y[     self.train_y==1     ].shape[0]/self.train_y.shape[0]
        stats['True Train Positive Ratio']  = self.train_true_y[self.train_true_y==1].shape[0]/self.train_true_y.shape[0]

        stats['Train Embedding Accuracy']   = self.train_y[self.train_y == self.train_true_y].shape[0]/self.train_y.shape[0]
        stats['Test Embedding Accuracy']    = self.test_y[ self.test_y == self.test_true_y  ].shape[0]/self.test_y.shape[0]

        # relevant features
        stats['# Relevant Features'] = self.explanation[self.explanation['scores'] != 0].shape[0]

        return stats




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
            return predict

        y = x_info.apply(get_embed_y, axis=1)

        self.regression_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=8)
        self.regression_model.fit(x, y)
        # Get the features of the test example
        test_index = self.test_data.index[(self.test_data['head'] == head) & (self.test_data['tail'] == tail)]
        test_x = self.test_x[test_index, :]
        test_y = self.test_data.iloc[test_index].apply(get_embed_y, axis=1)
        prediction = self.regression_model.predict(test_x)[:, 1]
        print "The triple has been predicted as ", prediction, " when should have been ", test_y




    def train(self):
        """ Train the explainable model.
        """
        self.grid_search.fit(self.train_x, self.train_y)

        # best model is accessed through `best_estimator_`
        self.model = self.grid_search.best_estimator_




    def explain_per_example(self, output_path, data_type, n_examples=10):
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
        examples_df = pd.DataFrame(explanations, columns=self.feature_names)
        examples_df.columns = self.feature_names

        final_reasons = examples_df.apply(get_reasons, axis=1)
        final_reasons['head'] = data.iloc[index]['head'].values
        final_reasons['tail'] = data.iloc[index]['tail'].values
        answers = self.model.predict_proba(features)[:, 1]
        final_reasons['y_logit'] = answers
        final_reasons['y_hat'] = y_hat
        final_reasons['y'] = y
        final_reasons.to_csv(output_path  + '/' + self.target_relation + '/' + self.target_relation + '.csv')
        return final_reasons

    def explain(self):
        """ Explain the model using the coefficients """
        # Extract the coefficients
        self.coefficients = self.model.coef_.reshape(-1,1)

        self.explanation = pd.DataFrame(self.coefficients, columns=['scores'])
        self.explanation['path'] = self.feature_names
        self.explanation = self.explanation.sort_values(by="scores", ascending=False)
        explanation = self.explanation[self.explanation['scores'] != 0]
        self.most_relevant_variables = pd.concat([explanation.iloc[0:10], explanation.iloc[-10:-1]])



def main(emb_model_path, feat_data_dir, corrupted_data_path):
    # parser = argparse.Argumentparser()

    # parser.add_argument(
    #     '--output', '-o',
    #     type=str,
    #     default='stats',
    #     help=''''stats' for relation statistics,
    #     'per-example' for per-example explanations '''
    # )

    # args = parser.parse_args()


    # data_base_name = 'FB13'

    ### Example of input variables to serve as reference later
    ## data_path = './results/NELL186/TransE/1524632595/pra_explain/results/g_hat_5nn_5negrate_bern'
    ## original_data_path = './benchmarks/NELL186'
    ## corrupted_data_path = './benchmarks/NELL186/corrupted/train2id_bern_2to1.txt'
    ## target_relations = os.listdir(data_path)

    # emb_model_path = './results/NELL186/TransE/1524632595/'
    # feat_data_dir = 'g_hat_5nn_5negrate_bern'
    # corrupted_data_path = ???
    data_path   = emb_model_path + '/pra_explain/results/'           + feat_data_dir # hardcoded, for now extracted features will always be in `pra_explain/results`
    output_path = emb_model_path + '/pra_explain/results_explained/' + feat_data_dir

    target_relations = os.listdir(data_path)
    model_info = pd.read_csv(emb_model_path + '/model_info.tsv', sep='\t')
    dataset_name = model_info['dataset_name']
    ground_truth_dataset_path = './benchmarks/' + model_info['dataset_name']





    relations_info = []
    target_relations = ['nationality'] # for debugging purposes
    for target_relation in target_relations:
        print("Training on " + target_relation + " relations")
        exp = Explanator(emb_model_path, ground_truth_dataset_path, target_relation, data_path)
        if exp.read_data():
            exp.train_local_regression("john_forsyth", "roman_empire")
            # exp.train():
            # print('    Generating explanation')
            # exp.explain()
            # print('    Generating report')
            # exp.report()
            # print('    Saving to csv')
            # exp.append_to_dataframe()
            ### # exp.export_dataframe(data_path + data_base_name + '.csv')
            relations_info.append(exp.get_stats())
            # print('    Generating per-example explanations')
            # exp.explain_per_example(data_path, 'test')
        else:
            print("No test data for relation `{}`.".format(target_relation))

    # save relations info somewhere
    pd.DataFrame(relations_info).to_csv(output_path + 'overall_info.tsv', sep='\t')
