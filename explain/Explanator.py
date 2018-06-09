from __future__ import division
import os
import time
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import pdb
from scipy.sparse import vstack
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier, LinearRegression, ElasticNet
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm

from tools import dataset_tools, train_test
from models import *
from config import Config
# from tools.feature_matrices import parse_feature_matrix

from helpers import parse_feature_matrix, getattr_else_None, get_dirs, ensure_dir, ensure_parentdir, get_reasons

### --------------------------------------------------------------------------
### --------------------------------------------------------------------------

class Explanator(object):
    def __init__(self, emb_model_path, ground_truth_dataset_path, n_jobs=1, max_knn_k=100):
        self.emb_model_path = emb_model_path
        self.ground_truth_dataset_path = ground_truth_dataset_path
        self.n_jobs = n_jobs
        self.max_knn_k = max_knn_k
        self.param_grid_logit = [{
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'alpha': [0.01, 0.001, 0.0001],
            'loss': ["log"],
            'penalty': ["elasticnet"],
            'max_iter': [100000],
            'tol': [1e-3],
            'class_weight': ["balanced"],
            'n_jobs': [n_jobs]
        }]
        self.param_grid_regression = [{
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'alpha': [0.1, 0.3, 0.9],
            'fit_intercept': [True],
            'max_iter': [10000],
            'normalize': [False],
        }]
        self.entity2id, self.id2entity = dataset_tools.read_name2id_file(os.path.join(ground_truth_dataset_path,'entity2id.txt'))
        self.relation2id, self.id2relation = dataset_tools.read_name2id_file(os.path.join(ground_truth_dataset_path, 'relation2id.txt'))
        # get the embedding model
        if not hasattr(self, 'emb_model'):
            self.emb_model = train_test.restore_model(self.emb_model_path)
            self.emb_model.calculate_thresholds()


    def emb_predict(self, heads, tails, rels, batch_size=10000):
        """Get the score for each head, tail and relation. Inputs `heads`, `tails`, and `rels`
        should be names, and not IDs.
        """
        # get head, tail and rel IDs
        heads2id = [self.entity2id[h]   for h in heads]
        tails2id = [self.entity2id[t]   for t in tails]
        rels2id  = [self.relation2id[r] for r in rels ]

        total_iters = (len(heads2id) + batch_size - 1) // batch_size # trick to ceil division using floor division

        scores = []
        for i in range(total_iters):
            start = (i) * batch_size
            end = (i+1) * batch_size
            res = self.emb_model.test_step(heads2id[start:end], tails2id[start:end], rels2id[start:end])
            scores = np.concatenate((scores, res), axis=0)
        return scores


    def predict(self, x):
        """Returns a prediction from the explainable model.
        """
        if self.model_name == 'global_logit' or self.model_name == 'local_logit':
            return self.model.predict(x)
        else:
            res = self.model.predict(x) > self.target_relation_thres
            return [1 if s else -1 for s in res]


    def get_kneighbors(self, ent):
        """Returns a (distance, indices) tuple for the k nearest neighbors of an entity, in the
        embedding vector space.

        Arguments:
        - `ent`: (string) entity name
        """
        # create knn instance if necessary
        if not hasattr(self, 'nbrs'):
            self.embed_params = self.emb_model.get_parameters()
            self.nbrs = NearestNeighbors(n_neighbors=self.max_knn_k, n_jobs=self.n_jobs).fit(self.embed_params['ent_embeddings'])

        # @TODO: check if neighbors have already been found for this entity and then save the results in memory
        return self.nbrs.kneighbors(self.embed_params['ent_embeddings'][self.entity2id[ent]].reshape(1, -1)) # @TODO: check if this is working


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
        if not os.path.exists(test_fpath) or os.stat(test_fpath).st_size == 0:
            print("There is no test file for relation `{}`, skipping.".format(target_relation))
            return False
        if not os.path.exists(train_fpath) or os.stat(train_fpath).st_size == 0:
            # raise IOError('`train.tsv` not present for relation `{}`'.format(target_relation))
            print("There is no train data for relation `{}`, skipping.".format(target_relation))
            return False

        # read train data (always present - not entirely true for NELL)
        self.train_heads, self.train_tails, self.train_y, train_feat_dicts = parse_feature_matrix(train_fpath)
        v = DictVectorizer(sparse=True)
        v.fit(train_feat_dicts)
        self.train_x = v.transform(train_feat_dicts)
        self.feature_names = v.get_feature_names()

        # read valid data (may not be present)
        if os.path.exists(valid_fpath) and os.stat(valid_fpath).st_size != 0:
            valid_heads, valid_tails, valid_y, valid_feat_dicts = parse_feature_matrix(valid_fpath)
            valid_x = v.transform(valid_feat_dicts)
            # we merge validation with training data, because the GridSearchCV creates the valid split automatically
            self.train_heads = np.concatenate((self.train_heads, valid_heads))
            self.train_tails = np.concatenate((self.train_tails, valid_tails))
            self.train_y     = np.concatenate((self.train_y,     valid_y    ))
            self.train_x     = vstack((self.train_x, valid_x)) # concatenate the sparse matrices vertically
            assert(self.train_y.shape[0] == self.train_x.shape[0])

        # read test data (always present)
        self.test_heads, self.test_tails, self.test_y, test_feat_dicts = parse_feature_matrix(test_fpath)
        self.test_x = v.transform(test_feat_dicts)

        # check that there are not only positive or negative labels in training set
        if len(np.unique(self.train_y)) <= 1:
            print("Not possible to train explainable model in relation `{}` because training set contains a single class.".format(target_relation))
            return False
        else:
            for class_ in np.unique(self.train_y):
                if len(self.train_y[self.train_y==class_]) < 3:
                    print("Not possible to train explainable model in relation `{}` because training set contains too few examples for one of the classes.".format(target_relation))
                    return False


        self.load_ground_truth_labels(self.ground_truth_dataset_path, target_relation)

        return True


    def load_ground_truth_labels(self, dataset_path, target_relation):
        """Read ground truth data from the original dataset and extract labels for the data
        present in the split (i.e., data whose features are extracted and that is present in
        `self.train_x`, etc.).

        We get the original data from a list of positive triples in order to dispense with the need
        for having the corrupted data path (which may change from model to model).
        """
        # these files have no labels, they are all positive instances
        gt_train = pd.read_csv(os.path.join(dataset_path, 'train2id.txt'), skiprows=1, sep=' ', names=['head', 'tail', 'relation'])
        gt_valid = pd.read_csv(os.path.join(dataset_path, 'valid2id.txt'), skiprows=1, sep=' ', names=['head', 'tail', 'relation'])
        gt_test  = pd.read_csv(os.path.join(dataset_path, 'test2id.txt' ), skiprows=1, sep=' ', names=['head', 'tail', 'relation'])

        # merge train and validation data
        gt_train = pd.concat((gt_train, gt_valid))

        # add labels (all positive)
        gt_train['label'] = 1
        gt_test ['label'] = 1

        # get id of target relation
        target_relation_id = self.relation2id[target_relation]

        # filter data to get only triples whose relation is the target relation
        gt_train_filt = gt_train.loc[gt_train['relation'] == target_relation_id]
        gt_test_filt  = gt_test.loc[ gt_test['relation']  == target_relation_id]

        # drop relation column in ground truth data
        gt_train_filt = gt_train_filt.drop('relation', axis=1)
        gt_test_filt = gt_test_filt .drop('relation', axis=1)

        # map ids to entities in ground truth data
        gt_train_filt['head'] = gt_train_filt['head'].map(self.id2entity)
        gt_train_filt['tail'] = gt_train_filt['tail'].map(self.id2entity)
        gt_test_filt ['head'] = gt_test_filt['head'].map(self.id2entity)
        gt_test_filt ['tail'] = gt_test_filt['tail'].map(self.id2entity)

        # create dataframe from split data
        train_df = pd.DataFrame({'head': self.train_heads, 'tail': self.train_tails})
        test_df  = pd.DataFrame({'head': self.test_heads , 'tail': self.test_tails })

        # merge to incorporate ground truth labels in dataframe split data
        train_df_merged = train_df.merge(gt_train_filt, how='left', on=['head', 'tail']).fillna(-1) # all unseen examples are negative
        test_df_merged  = test_df .merge(gt_test_filt , how='left', on=['head', 'tail']).fillna(-1) # all unseen examples are negative

        # get labels from merged dfs
        self.train_true_y = np.array(train_df_merged['label'])
        self.test_true_y  = np.array(test_df_merged ['label'])


    def get_results(self):
        """Outputs a dict containing results of the current model (for the current relation).
        Ideally, this function should be called each time after a new model has been fit.
        """
        stats = {}
        stats['model_name'] = self.model_name

        # relation and data information
        stats['Relation'] = self.target_relation
        stats['# Triples Train'] = len(self.train_heads)
        stats['# Triples Test '] = len(self.test_heads )
        # stats['# Triples Valid'] = ??? # we are now using the CV's validation sets

        # model parameters
        stats['alpha']             = getattr_else_None(self.model, 'alpha'            ) # @TODO: check if this works
        stats['l1_ratio']          = getattr_else_None(self.model, 'l1_ratio'         )
        stats['loss']              = getattr_else_None(self.model, 'loss'             )
        stats['penalty']           = getattr_else_None(self.model, 'penalty'          )
        stats['max_iter']          = getattr_else_None(self.model, 'max_iter'         )
        stats['tol']               = getattr_else_None(self.model, 'tol'              )
        stats['class_weight']      = getattr_else_None(self.model, 'class_weight'     )
        stats['fit_intercept']     = getattr_else_None(self.model, 'fit_intercept'    )
        stats['normalize']         = getattr_else_None(self.model, 'normalize'        )
        stats['n_nearby_examples'] = getattr_else_None(self,       'n_nearby_examples')

        # accuracy
        stats['Test Accuracy']              = self.model.score(self.test_x,  self.test_y)
        stats['True Test Accuracy']         = self.model.score(self.test_x,  self.test_true_y)
        stats['Train Accuracy']             = self.model.score(self.train_x, self.train_y)
        stats['True Train Accuracy']        = self.model.score(self.train_x, self.train_true_y)

        # precision
        stats['Test Precision']             = precision_score(self.test_y,       self.predict(self.test_x))
        stats['True Test Precision']        = precision_score(self.test_true_y,  self.predict(self.test_x))
        stats['Train Precision']            = precision_score(self.train_y,      self.predict(self.train_x))
        stats['True Train Precision']       = precision_score(self.train_true_y, self.predict(self.train_x))

        # recall
        stats['Test Recall']                = recall_score(self.test_y,       self.predict(self.test_x))
        stats['True Test Recall']           = recall_score(self.test_true_y,  self.predict(self.test_x))
        stats['Train Recall']               = recall_score(self.train_y,      self.predict(self.train_x))
        stats['True Train Recall']          = recall_score(self.train_true_y, self.predict(self.train_x))

        # F1 score
        stats['Test F1_score']              = f1_score(self.test_y,       self.predict(self.test_x))
        stats['True Test F1_score']         = f1_score(self.test_true_y,  self.predict(self.test_x))
        stats['Train F1_score']             = f1_score(self.train_y,      self.predict(self.train_x))
        stats['True Train F1_score']        = f1_score(self.train_true_y, self.predict(self.train_x))

        stats['Test Positive Ratio']        = self.test_y[      self.test_y==1      ].shape[0]/self.test_y.shape[0]
        stats['True Test Positive Ratio']   = self.test_true_y[ self.test_true_y==1 ].shape[0]/self.test_true_y.shape[0]
        stats['Train Positive Ratio']       = self.train_y[     self.train_y==1     ].shape[0]/self.train_y.shape[0]
        stats['True Train Positive Ratio']  = self.train_true_y[self.train_true_y==1].shape[0]/self.train_true_y.shape[0]

        stats['Train Embedding Accuracy']   = self.train_y[self.train_y == self.train_true_y].shape[0]/self.train_y.shape[0]
        stats['Test Embedding Accuracy']    = self.test_y[ self.test_y == self.test_true_y  ].shape[0]/self.test_y.shape[0]

        # relevant features
        stats['# Relevant Features'] = self.explanation[self.explanation['weight'] != 0].shape[0]
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
        self.target_relation_thres = self.emb_model.get_threshold_for_relation(self.relation2id[self.target_relation])

        # train regression
        self.model = LinearRegression()
        self.model.fit(self.train_x, self.train_y_scores)
        self.model_name = 'global_regression'

    def train_local_logit_for_all(self, output_path, get_local_data_func):
        results = pd.DataFrame(columns=['head', 'tail','y', 'y_hat', 'prediction'])
        for head, tail in tqdm(zip(self.test_heads, self.test_tails)):
            result = self.train_local_logit(output_path, head, tail, get_local_data_func)
            if result:
                results = results.append(result, ignore_index=True)
        output_filepath = os.path.join(output_path, self.target_relation, 'local_stats_logit' + '.tsv')
        ensure_parentdir(output_filepath)
        results.to_csv(output_filepath, sep='\t')

    def train_local_logit(self, output_path, head, tail, get_local_data_func, output_explanation=False):
        """Train a logistic regression model locally for the current relation and head and tail entities.
        """
        local_data = get_local_data_func(self, head, tail, y_type='labels')
        # Check if there are enough samples
        if local_data['x'].shape[0] < 10:
            return None

        # train local logit
        gs = GridSearchCV(SGDClassifier(), self.param_grid_logit, n_jobs=self.n_jobs)
        gs.fit(local_data['x'], local_data['y'])
        self.model = gs.best_estimator_
        self.model_name = 'local_logit'

        # Get the features of the test example
        test_index = local_data['index']
        test_x = self.test_x[test_index]
        test_y = self.test_y[test_index]
        prediction = self.model.predict(test_x)
        if output_explanation:
            self.explain_single_example(output_path, test_x, self.model.coef_, self.test_heads[test_index], self.test_tails[test_index], prediction, test_y, self.test_true_y[test_index])
        return {'head': head,
                'tail': tail,
                'y': self.test_true_y[test_index][0],
                'y_hat': test_y[0],
                'prediction': prediction[0]}

    def train_local_regression_for_all(self, output_path, get_local_data_func):
        results = pd.DataFrame()
        for head, tail in tqdm(zip(self.test_heads, self.test_tails)):
            results.append(self.train_local_regression(output_path, head, tail, get_local_data_func), ignore_index=True)
        output_filepath = os.path.join(output_path, self.target_relation, 'local_stats_regression' + '.tsv')
        ensure_parentdir(output_filepath)
        results.to_csv(output_filepath, sep='\t')

    def train_local_regression(self, output_path, head, tail, get_local_data_func, output_explanation=False):
        """Train a linear regression model locally for the current relation and head and tail entities.
        """
        local_data = get_local_data_func(self, head, tail, y_type='scores')
        self.target_relation_thres = self.emb_model.get_threshold_for_relation(self.relation2id[self.target_relation])

        # train regression
        self.model = LinearRegression()
        self.model.fit(local_data['x'], local_data['y'])
        self.model_name = 'local_regression'

        # Get the features of the test example
        test_index = local_data['index']
        test_x = self.test_x[test_index]
        test_y = self.test_y[test_index]
        prediction = self.model.predict(test_x)
        print "The triple has been predicted as ", prediction, " when should have been ", test_y
        if output_explanation:
            self.explain_single_example(output_path, test_x, self.model.coef_, self.test_heads[test_index], self.test_tails[test_index], prediction, test_y, self.test_true_y[test_index])
        return {'head': head,
                'tail': tail,
                'y': self.test_true_y[test_index],
                'y_hat': test_y,
                'prediction': prediction}


    def explain_per_example(self, output_path, data_type='test', n_examples=100):
        coefficients = self.model.coef_.reshape(-1, 1)
        if data_type == 'train':
            x = self.train_x
            y_hat = self.train_y
            y = self.train_true_y
            heads = self.train_heads
            tails = self.train_tails
        else:
            x = self.test_x
            y_hat = self.test_y
            y = self.test_true_y
            heads = self.test_heads
            tails = self.test_tails
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
        final_reasons['head'] = heads[index]
        final_reasons['tail'] = tails[index]
        if hasattr(self.model, 'predict_proba'):
            answers = self.model.predict_proba(features)[:, 1]
        else:
            answers = self.model.predict(features)
        final_reasons['y_logit'] = answers
        final_reasons['y_hat'] = y_hat[index]
        final_reasons['y'] = y[index]
        final_reasons.to_csv(os.path.join(output_path, self.target_relation + '.tsv'), sep='\t')
        return final_reasons


    def explain_single_example(self, output_path, features, coefficients, head, tail, y, y_hat, prediction):
        features = features.todense()
        explanations = np.multiply(features, coefficients).reshape(1, -1)
        example_df = pd.DataFrame(explanations, columns=self.feature_names)
        final_reasons = example_df.apply(get_reasons, axis=1)
        final_reasons['head'] = head
        final_reasons['tail'] = tail
        final_reasons['y_logit'] = prediction
        final_reasons['y_hat'] = y_hat
        final_reasons['y'] = y
        final_reasons.to_csv(os.path.join(output_path, self.target_relation, head[0] + '_' + tail[0] + '.tsv'), sep='\t')


    def explain_model(self, top_n=10, output_path=None):
        """Explain the model using the coefficients (weights) that the linear/logistic regression
        associated to each feature. The explanations are stored in `self.explanation`, a pandas
        DataFrame with `feature` and `weight` columns, sorted from highest to lowest weight.

        If `output_path` is provided, then the method will export the DataFrame as tsv to a folder
        with the same name of the current model in `output_path`.
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
