import os, subprocess
import pandas as pd
import numpy as np
import config, models
import train_test, dataset_tools, pra_setup


def extract_features_for_explaining(import_path, neg_rate, bern, feature_extractors, cuda_device=0, use_ids=False,
        g_hat_info=None):
    """Extract features using PRA library.

    Arguments:
    - `neg_rate` (int): negative to positive ratio
    - `bern` (bool): Bernoulli distribution for corrupting triples
    - `features_extractors` (list): ['pra', 'onesided', and/or 'anyrel']
    - `use_ids` (bool): indicates whether we should extract features using entities/relations ids
    instead of names
    - `g_hat_info` (dict): see below

    If extracting features from a graph predicted by the embedding model, a dict should be passed
    in `g_hat_info` with the following key-value pairs:
    - `knn_k` (int): number of nearest neighbors used to generate graph input
    """
    model_info = train_test.read_model_info(import_path)

    dataset_path = './benchmarks/{}/'.format(model_info['dataset_name'])
    distribution = 'bern' if bern else 'unif'
    corrupted_filename = 'train2id_{}negrate_{}.txt'.format(neg_rate, distribution)
    corrupted_dirpath = dataset_path + '/corrupted/'
    corrupted_filepath = corrupted_dirpath + corrupted_filename
    graph_input_dirname = '/pra_graph_input/' if not use_ids else '/pra_graph_input2id/'
    pra_graph_input_dir = os.path.abspath(dataset_path + graph_input_dirname)
    pra_explain_path = os.path.abspath(import_path + '/pra_explain/')
    experiment_specs_path = pra_explain_path + '/experiment_specs/'
    g_type = 'g' if not use_ids else 'g2id'
    g_hat_flag = False
    if g_hat_info != None:
        g_hat_flag = True
        split_name = 'g_hat_{}negrate_{}'.format(neg_rate, distribution)
        g_hat_fname_ids   = 'positives2id_{}nn.tsv'.format(g_hat_info['knn_k'])
        g_hat_fname_names = 'positives_{}nn.tsv'.format(g_hat_info['knn_k'])
        g_hat_path_ids = os.path.abspath(import_path + '/g_hat/' + g_hat_fname_ids)
        g_hat_path_names = os.path.abspath(import_path + '/g_hat/' + g_hat_fname_names)
        pra_graph_input_dir = g_hat_path_names
        g_type = 'ghat'
        if use_ids:
            g_type = 'ghat2id'
            pra_graph_input_dir = g_hat_path_ids
    split_name = '{}_{}negrate_{}'.format(g_type, neg_rate, distribution)


    # Ensure dirs exist
    def ensure_dir(d):
        if not os.path.exists(d):
            os.makedirs(d)
    ensure_dir(pra_explain_path)
    ensure_dir(experiment_specs_path)

    # Handle feature extraction strings and split name
    feature_extractor_dict = {
        'pra': 'PraFeatureExtractor',
        'onesided': 'OneSidedPathAndEndNodeFeatureExtractor',
        'anyrel': 'AnyRelFeatureExtractor'
    }
    spec_name = split_name + '__'
    feat_list = []
    for feat in feature_extractors:
        spec_name += '_' + feat
        feat_list.append('"{}"'.format(feature_extractor_dict[feat]))
    feat_extractor_string = ','.join(feat_list)


    if not g_hat_flag:
        ## Create original graph input for PRA
        if not use_ids:
            pra_setup.create_graph_input(
                dataset_path,
                labels=['valid.txt', 'test.txt'], # folds whose last column is the label
                graph_input_dirname=graph_input_dirname
            )
        else:
            pra_setup.create_graph_input(
                dataset_path,
                names_fname=['train2id.txt', 'test2id.txt', 'valid2id.txt'],
                labels=[], # folds whose last column is the label
                sep=' ',
                skiprows=1,
                order=['head', 'tail', 'relation'],
                graph_input_dirname=graph_input_dirname,
                use_ids=use_ids,
            )
    else:
        pass # ideally we would call predict_G here, but we are doing it before calling this function
    

    ## Generate/Read Negative Examples
    if not os.path.exists(corrupted_filepath):
        # create corrupted dirpath if not exist
        if not os.path.exists(corrupted_dirpath):
            os.makedirs(corrupted_dirpath)
        # generate corrupted set and save to disk in `corrupted` folder
        corrupted = dataset_tools.generate_corrupted_training_examples(dataset_path,
                neg_proportion=neg_rate, bern=bern)
        train2id = pd.DataFrame(corrupted)
        train2id.to_csv(corrupted_filepath,
            columns=['head', 'tail', 'relation', 'label'], index=False, header=False, sep=' ')
        print('Created corrupted file: {}.'.format(corrupted_filepath))
    else:
        train2id = pd.read_csv(corrupted_filepath,
            names=['head', 'tail', 'relation', 'label'], sep=' ', skiprows=0)
        print('Corrupted file already exists: {}.'.format(corrupted_filepath))

    ## Read validation and test examples

    valid2id_pos = pd.read_csv(dataset_path + 'valid2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    valid2id_neg = pd.read_csv(dataset_path + 'valid2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    test2id_pos = pd.read_csv(dataset_path + 'test2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    test2id_neg = pd.read_csv(dataset_path + 'test2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])

    valid2id_pos['label'] = 1
    valid2id_neg['label'] = -1
    test2id_pos['label'] = 1
    test2id_neg['label'] = -1

    valid2id = pd.concat((valid2id_pos, valid2id_neg))
    test2id = pd.concat((test2id_pos, test2id_neg))


    ## Restore working model

    con = train_test.restore_model(import_path)

    ## Predict and Update Data

    for fold in [train2id, valid2id, test2id]:
        fold['label'] = con.classify(fold['head'], fold['tail'], fold['relation'])
        fold['label'] = fold['label'].map(lambda x: 1 if x==1 else -1)

    entity2id, id2entity = dataset_tools.read_name2id_file(dataset_path + 'entity2id.txt')
    relation2id, id2relation = dataset_tools.read_name2id_file(dataset_path + 'relation2id.txt')
    n_relations = len(relation2id)

    ## Decode from id to names if necessary
    def decode_from_id_to_names(folds):
        for fold in folds:
            fold['head'] = fold['head'].map(id2entity)
            fold['tail'] = fold['tail'].map(id2entity)
            fold['relation'] = fold['relation'].map(id2relation)

    if not use_ids:
        decode_from_id_to_names([train2id, valid2id, test2id])

        if g_hat_flag:
            # convert 2ids g_hat file to names file
            g_hat_df = pd.read_csv(g_hat_path_ids, names=['head', 'relation', 'tail'], sep='\t', skiprows=0)
            decode_from_id_to_names([])
            g_hat_df.to_csv(g_hat_path_names, header=False, index=False, sep='\t', columns=['head', 'relation', 'tail'])

        # WARNING: at this stage we have transformed the dataframes,
        #   and entities and relations are not represented by ids anymore

    # Create Split
    pra_setup.create_split(
            {'train': train2id, 'valid': valid2id, 'test': test2id},
            splits_dirpath=import_path+'/pra_explain/splits',
            split_name=split_name)

    ## Setup PRA Experiment Specs
    if not g_hat_flag:
        spec = """
        {{
            "graph": {{
                "name": "{}",
                "relation sets": [
                    {{
                        "is kb": false,
                        "relation file": "{}/train.tsv"
                    }},
                    {{
                        "is kb": false,
                        "relation file": "{}/valid.tsv"
                    }}
                ]
            }},
            "split": "{}",
            "operation": {{
                "type": "create matrices",
                "features": {{
                    "type": "subgraphs",
                    "path finder": {{
                        "type": "BfsPathFinder",
                        "number of steps": 2
                    }},
                    "feature extractors": [
                        {}
                    ],
                    "feature size": -1
                }}
            }},
            "output": {{ "output matrices": true }}
        }}

        """.format(g_type, pra_graph_input_dir, pra_graph_input_dir, split_name, feat_extractor_string)
    else:
        spec = """
        {{
            "graph": {{
                "name": "g_hat_{}nn",
                "relation sets": [
                    {{
                        "is kb": false,
                        "relation file": "{}"
                    }}
                ]
            }},
            "split": "{}",
            "operation": {{
                "type": "create matrices",
                "features": {{
                    "type": "subgraphs",
                    "path finder": {{
                        "type": "BfsPathFinder",
                        "number of steps": 2
                    }},
                    "feature extractors": [
                        {}
                    ],
                    "feature size": -1
                }}
            }},
            "output": {{ "output matrices": true }}
        }}

        """.format(g_hat_info['knn_k'], pra_graph_input_dir, split_name, feat_extractor_string)
    
    spec_fpath = '{}/experiment_specs/{}.json'.format(pra_explain_path, spec_name)
    with open(spec_fpath, 'w') as f:
        f.write(spec)
    print "Spec file written: {}".format(spec_fpath)


    ## Extract Features
    bash_command = '/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/tools/run_pra_n_times.sh {} {} {}'\
        .format(pra_explain_path, spec_name, n_relations)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output, error
    print("\nIf PRA run successfully, features were extracted and saved into `{}`.".format(pra_explain_path))


def extract_features_from_dataset(dataset_path, neg_rate, bern, feature_extractors, use_ids=False):
    """Extracts features from a dataset. This is useful when one wants to run PRA directly on an input
    graph.
    """
    pra_path = os.path.abspath(dataset_path + '/pra')
    distribution = 'bern' if bern else 'unif'
    corrupted_filename = 'train2id_{}negrate_{}.txt'.format(neg_rate, distribution)
    corrupted_dirpath = dataset_path + '/corrupted/'
    corrupted_filepath = corrupted_dirpath + corrupted_filename
    split_name = 'original_data' if not use_ids else 'original_data2id'
    pra_graph_input_dir = dataset_path + '/pra_graph_input/' if not use_ids else dataset_path + '/pra_graph_input2id/'
    pra_graph_input_dir = os.path.abspath(pra_graph_input_dir)
    g_type = 'g' if not use_ids else 'g2id'

    # Handle feature extraction strings and split name
    feature_extractor_dict = {
        'pra': 'PraFeatureExtractor',
        'onesided': 'OneSidedPathAndEndNodeFeatureExtractor',
        'anyrel': 'AnyRelFeatureExtractor'
    }
    spec_name = split_name + '__'
    feat_list = []
    for feat in feature_extractors:
        spec_name += '_' + feat
        feat_list.append('"{}"'.format(feature_extractor_dict[feat]))
    feat_extractor_string = ','.join(feat_list)

    ensure_dir(pra_path)
    ensure_dir(pra_path + '/experiment_specs')
    
    spec = """
    {{
        "graph": {{
            "name": "{}",
            "relation sets": [
                {{
                    "is kb": false,
                    "relation file": "{}/train.tsv"
                }},
                {{
                    "is kb": false,
                    "relation file": "{}/valid.tsv"
                }}
            ]
        }},
        "split": "{}",
        "operation": {{
            "type": "create matrices",
            "features": {{
                "type": "subgraphs",
                "path finder": {{
                    "type": "BfsPathFinder",
                    "number of steps": 2
                }},
                "feature extractors": [
                    {}
                ],
                "feature size": -1
            }}
        }},
        "output": {{ "output matrices": true }}
    }}
    """.format(g_type, pra_graph_input_dir, pra_graph_input_dir, split_name, feat_extractor_string)
    spec_fpath = '{}/experiment_specs/{}.json'.format(pra_path, spec_name)
    with open(spec_fpath, 'w') as f:
        f.write(spec)
    print "Spec file written: {}".format(spec_fpath)



    ## Generate/Read Negative Examples
    if not os.path.exists(corrupted_filepath):
        # create corrupted dirpath if not exist
        if not os.path.exists(corrupted_dirpath):
            os.makedirs(corrupted_dirpath)
        # generate corrupted set and save to disk in `corrupted` folder
        corrupted = dataset_tools.generate_corrupted_training_examples(dataset_path,
                neg_proportion=neg_rate, bern=bern)
        train2id = pd.DataFrame(corrupted)
        train2id.to_csv(corrupted_filepath,
            columns=['head', 'tail', 'relation', 'label'], index=False, header=False, sep=' ')
        print('Created corrupted file: {}.'.format(corrupted_filepath))
    else:
        train2id = pd.read_csv(corrupted_filepath,
            names=['head', 'tail', 'relation', 'label'], sep=' ', skiprows=0)
        print('Corrupted file already exists: {}.'.format(corrupted_filepath))

    ## Read validation and test examples

    valid2id_pos = pd.read_csv(dataset_path + '/valid2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    valid2id_neg = pd.read_csv(dataset_path + '/valid2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    test2id_pos = pd.read_csv(dataset_path + '/test2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    test2id_neg = pd.read_csv(dataset_path + '/test2id_neg.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])

    valid2id_pos['label'] = 1
    valid2id_neg['label'] = -1
    test2id_pos['label'] = 1
    test2id_neg['label'] = -1

    valid2id = pd.concat((valid2id_pos, valid2id_neg))
    test2id = pd.concat((test2id_pos, test2id_neg))
    
    
    # get data
    entity2id, id2entity = dataset_tools.read_name2id_file(dataset_path + '/entity2id.txt')
    relation2id, id2relation = dataset_tools.read_name2id_file(dataset_path + '/relation2id.txt')
    n_relations = len(relation2id)

    ## Decode from id to names if necessary
    def decode_from_id_to_names(folds):
        for fold in folds:
            fold['head'] = fold['head'].map(id2entity)
            fold['tail'] = fold['tail'].map(id2entity)
            fold['relation'] = fold['relation'].map(id2relation)

    if not use_ids:
        decode_from_id_to_names([train2id, valid2id, test2id])

        # WARNING: at this stage we have transformed the dataframes,
        #   and entities and relations are not represented by ids anymore


    # Create Split
    pra_setup.create_split(
            {'train': train2id, 'valid': valid2id, 'test': test2id},
            splits_dirpath=pra_path+'/splits',
            split_name=split_name)
    
    ## Extract Features
    bash_command = '/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/tools/run_pra_n_times.sh {} {} {}'\
        .format(pra_path, spec_name, n_relations)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output, error
    print("\nIf PRA run successfully, features were extracted and saved into `{}`.".format(pra_path))
    

def ensure_dir(dirpath):
    """Creates the directory if it does not exist.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
