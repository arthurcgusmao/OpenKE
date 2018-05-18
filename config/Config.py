#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
from ctypes import c_float # needed to extract the average accuracy value
import json
import models

class Config(object):

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
        self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.test_flag = False
        self.in_path = "./"
        self.out_path = None
        self.bern = 0
        self.hidden_size = 100
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.train_times = 0
        self.margin = 1.0
        self.nbatches = 100
        self.batch_size = None
        self.negative_ent = 1
        self.negative_rel = 0
        self.workThreads = 1
        self.alpha = 0.001
        self.lmbda = 0.000
        self.log_on = 1
        self.log_type = 'epoch'
        self.log_print = True
        self.lr_decay=0.000
        self.weight_decay=0.000
        self.exportName = None
        self.importName = None
        self.export_steps = 0
        self.opt_method = "SGD"
        self.optimizer = None
        self.test_link_prediction = False
        self.test_triple_classification = False
        self.log = {} # logging dict where we'll save information about training/testing
        self.shuffle = 1 # shuffle the training set for each epoch (instead of randomly sampling from it)
        # start tf operations to be run for the classify function --------
        self.classify_scores = None
        self.classify_relations = None
        self.classify_classes = None
        # end tf operations to be run for the classify function --------
    def init(self):
        self.trainModel = None
        if self.in_path != None:
            self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
            self.lib.setBern(self.bern)
            self.lib.setShuffle(self.shuffle)
            self.lib.setWorkThreads(self.workThreads)
            self.lib.randReset()
            self.lib.importTrainFiles()
            self.update_batch_size_and_nbatches()
            self.relTotal = self.lib.getRelationTotal()
            self.entTotal = self.lib.getEntityTotal()
            self.trainTotal = self.lib.getTrainTotal()
            self.testTotal = self.lib.getTestTotal()
            self.validTotal = self.lib.getValidTotal()
            self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
            self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
            self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
            self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
            self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
            self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
            self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
            self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
            self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
        if self.test_link_prediction:
            self.lib.importTestFiles()
            self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
            self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
            self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
            self.test_h_addr = self.test_h.__array_interface__['data'][0]
            self.test_t_addr = self.test_t.__array_interface__['data'][0]
            self.test_r_addr = self.test_r.__array_interface__['data'][0]
        if self.test_triple_classification:
            self.lib.importTestFiles()
            self.lib.importTypeFiles()

            self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
            self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
            self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
            self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
            self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
            self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
            self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

            self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
            self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
            self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
            self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
            self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
            self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
            self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]

    def get_ent_total(self):
        return self.entTotal

    def get_rel_total(self):
        return self.relTotal

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_opt_method(self, method):
        self.opt_method = method

    def set_test_link_prediction(self, flag):
        self.test_link_prediction = flag

    def set_test_triple_classification(self, flag):
        self.test_triple_classification = flag

    def set_log_on(self, flag, log_type='epoch', log_print=True):
        self.log_on = flag
        self.log_type = log_type
        self.log_print = log_print

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_in_path(self, path):
        self.in_path = path

    def set_out_files(self, path):
        self.out_path = path

    def set_bern(self, bern):
        self.bern = bern

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        self.rel_size = dim

    def set_train_times(self, times):
        self.train_times = times

    def set_nbatches(self, nbatches):
        self.batch_size = None
        self.nbatches = nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = None

    def update_batch_size_and_nbatches(self):
        if self.batch_size == None:
            self.batch_size = (self.lib.getTrainTotal() + self.nbatches - 1) / self.nbatches # trick to ceil division using floor division
        elif self.batch_size > self.lib.getTrainTotal():
            self.batch_size = self.lib.getTrainTotal()
        self.nbatches = (self.lib.getTrainTotal() + self.batch_size - 1) / self.batch_size # trick to ceil division using floor division

    def set_margin(self, margin):
        self.margin = margin

    def set_work_threads(self, threads):
        self.workThreads = threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path):
        self.exportName = path

    def set_export_steps(self, steps):
        self.export_steps = steps

    def set_lr_decay(self,lr_decay):
        self.lr_decay=lr_decay

    def set_weight_decay(self,weight_decay):
        self.weight_decay=weight_decay

    def sampling(self):
        self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

    def save_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, self.exportName)

    def restore_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, self.importName)


    def export_variables(self, path = None):
        with self.graph.as_default():
            with self.sess.as_default():
                if path == None:
                    self.saver.save(self.sess, self.exportName)
                else:
                    self.saver.save(self.sess, path)

    def import_variables(self, path = None):
        with self.graph.as_default():
            with self.sess.as_default():
                if path == None:
                    self.saver.restore(self.sess, self.importName)
                else:
                    self.saver.restore(self.sess, path)

    def get_parameter_lists(self):
        return self.trainModel.parameter_lists

    def get_parameters_by_name(self, var_name):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    return self.sess.run(self.trainModel.parameter_lists[var_name])
                else:
                    return None

    def get_parameters(self, mode = "numpy"):
        res = {}
        lists = self.get_parameter_lists()
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = self.get_parameters_by_name(var_name)
            else:
                res[var_name] = self.get_parameters_by_name(var_name).tolist()
        return res

    def save_parameters(self, path = None):
        if path == None:
            path = self.out_path
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def set_parameters_by_name(self, var_name, tensor):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    self.trainModel.parameter_lists[var_name].assign(tensor).eval()

    def set_parameters(self, lists):
        for i in lists:
            self.set_parameters_by_name(i, lists[i])

    def set_parameters_from_json(self, filepath):
        with open(filepath) as f:
            embed = json.load(f)
            self.set_parameters(embed)

    def set_model(self, model, **kwargs):
        self.model = model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform = True)
                with tf.variable_scope("model", reuse=None, initializer = initializer):
                    self.trainModel = self.model(config = self)
                    if self.optimizer != None:
                        pass
                    elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
                        self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
                    elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
                        self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
                    elif self.opt_method == "Adam" or self.opt_method == "adam":
                        self.optimizer = tf.train.AdamOptimizer(self.alpha)
                    elif self.opt_method == "RMSProp" or self.opt_method == "rmsprop":
                        self.optimizer = tf.train.RMSPropOptimizer(self.alpha)
                    else:
                        self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
                    grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)
                self.saver = tf.train.Saver()
                self.sess.run(tf.global_variables_initializer())

    def set_model_by_name(self, model_name):
        self.set_model(getattr(models, model_name))

    def train_step(self, batch_h, batch_t, batch_r, batch_y):
        feed_dict = {
            self.trainModel.batch_h: batch_h,
            self.trainModel.batch_t: batch_t,
            self.trainModel.batch_r: batch_r,
            self.trainModel.batch_y: batch_y
        }
        _, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
        return loss

    def test_step(self, test_h, test_t, test_r):
        feed_dict = {
            self.trainModel.predict_h: test_h,
            self.trainModel.predict_t: test_t,
            self.trainModel.predict_r: test_r,
        }
        predict = self.sess.run(self.trainModel.predict, feed_dict)
        return predict

    def run(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()
                self.log['training_curve'] = []
                start_time = time.time()
                for epoch in range(self.train_times):
                    epoch_loss = 0.0
                    for batch in range(self.nbatches):
                        self.sampling()
                        batch_loss = self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
                        epoch_loss += batch_loss
                        # logging
                        if self.log_on and self.log_type == 'batch':
                            valid_acc = self.validation_acc()
                            self.log['training_curve'].append({'epoch': epoch,
                                                               'batch': batch,
                                                               'epoch_loss': epoch_loss,
                                                               'batch_loss': batch_loss,
                                                               'valid_acc': valid_acc})
                            if self.log_print:
                                print "Epoch: {:4d},\tBatch: {:3d},\tEpoch Loss: {:9.3f},\tBatch Loss: {:7.3f}\tValid Acc: {:0.4f}".format(epoch, batch, epoch_loss, batch_loss, valid_acc)
                    # printing and logging info
                    if self.log_on == 1 and self.log_type == 'epoch':
                        valid_acc = self.validation_acc()
                        self.log['training_curve'].append({'epoch': epoch,
                                                           'epoch_loss': epoch_loss,
                                                           'valid_acc': valid_acc})
                        if self.log_print:
                            print "Epoch: {:4d},\tEpoch Loss: {:9.3f},\tValid Acc: {:0.4f}".format(epoch, epoch_loss, valid_acc)
                    if self.exportName != None and (self.export_steps!=0 and epoch % self.export_steps == 0):
                        self.save_tensorflow()
                self.log['learning_time'] = time.time() - start_time
                if self.exportName != None:
                    self.save_tensorflow()
                if self.out_path != None:
                    self.save_parameters(self.out_path)

    def test(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()
                start_time = time.time()
                if self.test_link_prediction:
                    self.lib.initTestLinkPrediction()
                    total = self.lib.getTestTotal()
                    if self.log_on: tmp = -1
                    for times in range(total):
                        self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
                        res = self.test_step(self.test_h, self.test_t, self.test_r)
                        self.lib.testHead(res.__array_interface__['data'][0])

                        self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
                        res = self.test_step(self.test_h, self.test_t, self.test_r)
                        self.lib.testTail(res.__array_interface__['data'][0])
                        if self.log_on:
                            if times*10 / total > tmp:
                                print("Testing link prediction: {:3d}% ...".format(times*100 / total))
                                tmp = times*10 / total
                    if self.log_on:
                        print("Testing link prediction: 100% Done.\n".format(times*100 / total))
                    self.lib.test_link_prediction()
                if self.test_triple_classification:
                    self.calculate_thresholds()
                    self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)
                    res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
                    res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
                    self.lib.test_triple_classification(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
                self.log['testing_time'] = time.time() - start_time


    def calculate_thresholds(self):
        self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
        res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
        res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
        self.lib.getBestThreshold(res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

    def validation_acc(self):
        """Returns the validation set accuracy for the best threshold.
        """
        self.calculate_thresholds()
        valid_acc = c_float.in_dll(self.lib, 'validAcc').value
        return valid_acc

    def get_threshold_for_relation(self, r):
        """Returns the optimal threshold (found using the validation set) for a specific relation.
        """
        self.lib.update_threshold_for_relation(r)
        return c_float.in_dll(self.lib, 'threshold_for_relation').value

    # def get_threshold_dict_for_relations(self, rels):
    #     """Returns a dict whose keys are relation indexes and values are the respective threshold.
    #
    #     Arguments:
    #     - rels: a numpy array of relations.
    #     """
    #     self.calculate_thresholds()
    #     unique_rels = np.unique(rels)
    #     thres_dict = {}
    #     for r in unique_rels:
    #         thres_dict[r] = self.get_threshold_for_relation(r)
    #     return thres_dict

    def get_threshold_list_for_relations(self, rels):
        """Returns a dict whose keys are relation indexes and values are the respective threshold.

        Arguments:
        - rels: a numpy array of relations.
        """
        self.calculate_thresholds()
        unique_rels = np.unique(rels)
        thres_list = []
        for r in unique_rels:
            thres_list.append(self.get_threshold_for_relation(r))
        return thres_list


    def setup_classify_graph(self, rels):
        """Returns the classification of a set of triples, using the validation threshold."""
        thres_list = self.get_threshold_list_for_relations(rels)
        thres_params = tf.constant(thres_list)

        self.classify_scores = tf.placeholder(tf.float32, [None])
        self.classify_relations = tf.placeholder(tf.int64, [None])
        thresholds = tf.gather(thres_params, self.classify_relations)
        self.classify_classes = tf.less(self.classify_scores, thresholds)


    def classify(self, heads, tails, rels, batch_size=10000, update_thres=False):
        """Runs the graph created by classify setup.
        """
        if (self.classify_classes == None) or (update_thres):
            self.setup_classify_graph(rels)

        if len(rels) % batch_size == 0:
            total_iters = len(rels) / batch_size
        else:
            total_iters = (len(rels) / batch_size) + 1

        output = np.array([])
        with tf.Session() as sess:
            for i in range(total_iters):
                print("Classifying iteration {} of {}".format(i, total_iters))
                start = (i) * batch_size
                end = (i+1) * batch_size
                res = tf.Session().run(self.classify_classes, feed_dict={
                    self.classify_scores: self.test_step(heads[start:end], tails[start:end], rels[start:end]).reshape(-1),
                    self.classify_relations: rels[start:end]
                })
                output = np.concatenate((output, res))
        return output
