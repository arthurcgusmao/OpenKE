from __future__ import division
import os
import time
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer

import config, models
from tools import dataset_tools, train_test
# from tools.feature_matrices import parse_feature_matrix

### --------------------------------------------------------------------------
### --------------------------------------------------------------------------

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

    return heads, tails, labels, feat_dicts


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


### --------------------------------------------------------------------------
### --------------------------------------------------------------------------

class Explanator(object):
    def __init__(self, emb_model_path, ground_truth_dataset_path, n_jobs=4):
        self.emb_model_path = emb_model_path
        self.ground_truth_dataset_path = ground_truth_dataset_path
        self.n_jobs = n_jobs
        self.param_grid_logit = [{
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'alpha': [0.01, 0.001, 0.0001],
            'loss': ["log"],
            'penalty': ["elasticnet"],
            'max_iter': [100000],
            'tol': [1e-3],
            'class_weight': ["balanced"],
            'n_jobs': [n_jobs],
        }]
        self.param_grid_regression = [{
            'fit_intercept': [True, False],
            'normalize': [False, True],
            'n_jobs': [n_jobs],
        }]
        self.entity2id, self.id2entity = dataset_tools.read_name2id_file(os.path.join(ground_truth_dataset_path,'entity2id.txt'))
        self.relation2id, self.id2relation = dataset_tools.read_name2id_file(os.path.join(ground_truth_dataset_path, 'relation2id.txt'))


    def emb_predict(heads, tails, rels, batch_size=10000):
        """Get the score for each head, tail and relation. Inputs `heads`, `tails`, and `rels`
        should be names, and not IDs.
        """
        # get the embedding model
        if not hasattr(self, 'emb_model'):
            self.emb_model = train_test.restore_model(self.emb_model_path)

        # get head, tail and rel IDs
        heads2id = [self.entity2id[h]   for h in heads]
        tails2id = [self.entity2id[t]   for t in tails]
        rels2id  = [self.relation2id[r] for r in rels ]

        total_iters = (len(heads2id) + batch_size - 1) // batch_size # trick to ceil division using floor division

        scores = []
        for i in range(total_iters):
            start = (i) * batch_size
            end = (i+1) * batch_size
            scores += self.emb_model.test_step(heads2id[start:end], tails2id[start:end], rels2id[start:end])
        return scores


    def define_knn(max_knn_k=100):
        # get the embedding model
        if not hasattr(self, 'emb_model'):
            self.emb_model = train_test.restore_model(self.emb_model_path)

        self.embed_params = self.emb_model.get_parameters()
        self.nbrs = NearestNeighbors(n_neighbors=max_knn_k, n_jobs=4).fit(self.embed_params['ent_embeddings'])


    def load_data(self, split_path, target_relation):
        """Read embedding predicted data for the target relation from the split (i.e., from data
        whose features were extracted and whose labels are predictions from the embedding model).

        -------------------------------------
        WARNING: this function should be called whenever we change from one relation to another, so
        that it changes the `target_relation` and get the split data for the new relation.
        -------------------------------------

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
        self.target_relation = target_relation

        train_fpath = "{}/{}/train.tsv".format(split_path, target_relation)
        valid_fpath = "{}/{}/valid.tsv".format(split_path, target_relation)
        test_fpath  = "{}/{}/test.tsv" .format(split_path, target_relation)

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
            self.train_y     += valid_y
            self.train_x      = vstack((self.train_x, valid_x)) # concatenate the sparse matrices vertically

        # read test data (always present)
        self.test_heads, self.test_tails, self.test_y, test_feat_dicts = parse_feature_matrix(test_fpath)
        self.test_x = v.transform(test_feat_dicts)

        # check that there are not only positive or negative labels in training set
        if len(np.unique(self.train_y)) <= 1:
            print("Not possible to train explainable model in relation `{}` because training set contains a single class.".format(target_relation))
            return False

        return True



    def load_ground_truth_labels(self, dataset_path, target_relation):
        """Read ground truth data from the original dataset and extract labels for the data
        present in the split (i.e., data whose features are extracted and that is present in
        `self.train_x`, etc.).

        We get the original data from a list of positive triples in order to dispense with the need
        for having the corrupted data path (that may change from model to model).
        """
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
        heads2id = [self.entity2id[h] for h in self.train_heads]
        tails2id = [self.entity2id[t] for t in self.train_tails]

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


    def get_results():
        """Outputs a dict containing results of the current model (for the current relation).
        Ideally, this function should be called each time after a new model has been fit.
        """
        stats = {}
        stats['model_name'] = self.model_name

        # relation and data information
        stats['Relation'] = self.target_relation
        stats['# Triples Train'] = len(train_heads)
        stats['# Triples Test '] = len(test_heads )
        # stats['# Triples Valid'] = ??? # we are now using the CV's validation sets

        # model parameters
        stats['alpha']    = self.model['alpha'   ] # @TODO: check if this works
        stats['l1_ratio'] = self.model['l1_ratio'] # @TODO: check if this works

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
        stats['# Relevant Features'] = self.explanation[self.explanation['weights'] != 0].shape[0]
        # NOTE: in the future this should be changed, just because a feature has weight different
        #       than zero it doesn't necessarily mean that it is relevant. We must find a way to
        #       define this "relevance" formally.

        return stats


    def train_global_logit(self):
        """Trains a logistic regression model globally for the current relation.
        """
        gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=self.n_jobs)
        gs.fit(self.train_x, self.train_y)
        self.model = gs.best_estimator_
        self.model_name = 'global_logit'


    def train_global_regression(self):
        """Trains a linear regression model globally for the current relation.
        """
        # get embedding scores (not labels)
        train_rels = [self.target_relation] * len(self.train_heads) # list of relations to be passed to `self.emb_predict()`
        self.train_y_scores = self.emb_predict(self.train_heads, self.train_tails, train_rels)

        gs = GridSearchCV(LinearRegression(copy_X=True), self.param_grid_regression, n_jobs=self.n_jobs)
        # self.model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=self.n_jobs)
        gs.fit(self.train_x, self.train_y_scores)
        self.model = gs.best_estimator_
        self.model_name = 'global_regression'



    def train_local_logit(self, head, tail):
        """Train and evaluate the model locally """
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

        y = x_info.apply(self.embed_predict, axis=1)

        self.regression_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=8)
        self.regression_model.fit(x, y)
        # Get the features of the test example
        test_index = self.test_data.index[(self.test_data['head'] == head) & (self.test_data['tail'] == tail)]
        test_x = self.test_x[test_index, :]
        test_y = self.test_data.iloc[test_index].apply(self.embed_predict, axis=1)
        prediction = self.regression_model.predict(test_x)[:, 1]
        print "The triple has been predicted as ", prediction, " when should have been ", test_y


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


    def explain_model(self, top_n=10, output_path=None):
        """Explain the model using the coefficients (weights) that the linear/logistic regression
        associated to each feature. The explanations is stores in `self.explanation`, a pandas
        DataFrame with `feature` and `weight` columns, sorted from highest to lowest weight.
        """
        self.explanation = pd.DataFrame({
            'weight': self.model.coef_.reshape(-1), # extract coefficients (weights)
            'feature':   self.feature_names,
        }).sort_values(by="weight", ascending=False)

        # remove features whose weight is zero
        filter = self.explanation[self.explanation['weight'] != 0]
        # get the top_n relevant features (for both positive and negative weights)
        self.top_n_relevant_features = pd.concat([filter.iloc[0:top_n], filter.iloc[-top_n:-1]])

        # save explanation if `output_path` provided
        if output_path:
            output_dir      = os.path.join(output_path, self.model_name)
            output_filepath = os.path.join(output_dir,  '{}.tsv'.format(self.target_relation))
            ensure_dir(output_dir)
            self.explanation.to_csv(output_filepath, sep='\t', columns=['weight', 'feature'], index=False)


### --------------------------------------------------------------------------
### --------------------------------------------------------------------------

def pipeline(emb_model_path, splits=None):
    """Runs a pipeline for producing explanations with different models for an embedding model.

    Arguments:
    - `emb_model_path`: (string) path to the embedding model directory.
    - `splits`: (list) directory names (inside `/pra_explain/results`) for which the pipeline
                       should be run.
    """
    # read model information
    model_info = pd.read_csv(emb_model_path + '/model_info.tsv', sep='\t')
    ground_truth_dataset_path = './benchmarks/' + model_info['dataset_name'].iloc[0]

    # define directory path variables
    pra_results_path  = emb_model_path + '/pra_explain/results/'
    expl_results_path = emb_model_path + '/pra_explain/results_explained/'
    ensure_dir(expl_results_path)

    # get a list of splits (different feature extractions, e.g., using G and G_hat) to run if not provided
    if splits == None:
        splits = get_dirs(pra_results_path)

    # instantiate Explanator for this model
    expl = Explanator(emb_model_path, ground_truth_dataset_path)

    for split in splits:

        split_path  = os.path.join(pra_results_path,  split)
        output_path = os.path.join(expl_results_path, split)
        ensure_dir(output_path)

        target_relations = get_dirs(split_path) # get a list of target relations
        results = []

        target_relations = ['nationality'] # for debugging purposes
        for target_relation in target_relations:
            print("Loading data for `{}`...".format(target_relation))

            if expl.load_data(split_path, target_relation):

                # global logit
                expl.train_global_logit()
                expl.explain_model(output_path=output_path)
                expl.explain_per_example(data_path, 'test')
                results.append(expl.get_results())

                # global regression
                expl.train_global_regression()
                # ...

                # local logit
                # expl.train_local_logit()
                # ...

                # local regression
                # expl.train_local_regression()
                # ...
            else:
                print("Could not load data for `{}`. Skipping relation...".format(target_relation))

        # save overall results
        pd.DataFrame(results).to_csv(output_path + '/overall_results.tsv', sep='\t')
