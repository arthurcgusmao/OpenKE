# --- start make file run from another dir ---
#
# Note: File should be run from XKE root dir. E.g.:
#
#       $ cd Projects/XKE
#       $ python examples/emb_grid_search/grid_search_TransE_FB13.py
#
import os, sys
xke_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, xke_root)
# --- end make file run from another dir ---


from tools import train_test

train_test.grid_search([
    {
        ### NECESSARY hyperparameters
        ### -------------------------
        'dataset_name'      : 'FB15k',
        'model_name'        : "Analogy",

        'batch_size'                 : [8192,1024],
        # 'n_batches'                  : 100, # number of batches
        'n_epochs'                   : 500, # epochs
        'learning_rate'              : [0.1,1.0,0.01],
        'k'                          : 200, # embedding dimension
        'opt_method'                 : 'adagrad',


        ### OPTIONAL or MODEL DEPENDENT hyperparameters
        ### -------------------------------------------
        # 'margin'                     : 1.0,  # for most models
        # 'score_norm'                 : 'l2', # for TransX models
        # 'embedding_initialization'   : 'TransE/1524490825', # initialize parameters from another model
        'regul_weight'               : [0.001, 0.01], # the regularization weight in the loss function


        ### WITH DEFAULT VALUE hyperparameters
        ### ----------------------------------

        ### negative examples and training settings
        'ent_neg_rate'               : 4, # (defaults to 1)
        'rel_neg_rate'               : 2, # (defaults to 0)
        'bern'                       : 0, # (defaults to 1) Bernoulli distribution for generating negative training examples
        # 'shuffle'                    : 1, # (defaults to 1) Shuffle training set (each epoch) instead of randomly sampling from it

        ### test settings
        # 'test_link_prediction'   : True, # (defaults to True)
        # 'test_triple_class'      : True, # (defautls to True)

        ### logging settings
        # 'log_on'         : 1, # (defaults to 1)
        # 'log_type'       : 'epoch', # (defaults to 'epoch')
        # 'log_print'      : True, # (defaults to True)

        ### GPU and CPU settings
        # 'work_threads'    : multiprocessing.cpu_count(), # (defaults to multiprocessing.cpu_count())
        'cuda_device'     : 1, # (no default value -- not necessary)
    },
])
