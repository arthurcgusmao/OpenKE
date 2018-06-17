import os
import pandas as pd

from tools import train_test
from helpers import ensure_dir, get_dirs
from Explanator import Explanator
from metrics import calc_metrics, calc_overall_metrics
import local_functions as lfs


def pipeline(emb_model_path, splits=None, adapt_run_sfe_wo_emb=False):
    """Runs a pipeline for producing explanations with different models for an embedding model.

    Arguments:
    - `emb_model_path`: (string) path to the embedding model directory.
    - `splits`: (list) directory names (inside `/pra_explain/results`) for which the pipeline
                       should be run.
    """
    if not adapt_run_sfe_wo_emb:
        # read model information
        model_info = pd.read_csv(emb_model_path + '/model_info.tsv', sep='\t')
        ground_truth_dataset_path = './benchmarks/' + model_info['dataset_name'].iloc[0]

        # define directory path variables
        pra_results_path  = emb_model_path + '/pra_explain/results/'
        expl_results_path = emb_model_path + '/pra_explain/results_explained/'
        ensure_dir(expl_results_path)
    else:
        # in this case emb_model_path is actually a dataset path
        ground_truth_dataset_path = emb_model_path
        pra_results_path  = emb_model_path + '/pra/results/'
        expl_results_path = emb_model_path + '/pra/results_explained/'
        ensure_dir(expl_results_path)

    # get a list of splits (different feature extractions, e.g., using G and G_hat) to run if not provided
    if splits == None:
        splits = get_dirs(pra_results_path)

    if not adapt_run_sfe_wo_emb:
        # instantiate Explanator for this model
        expl = Explanator(emb_model_path, ground_truth_dataset_path)
    else:
        expl = Explanator(None, ground_truth_dataset_path)

    for split in splits:
        print "\n####\n{}\n####".format(emb_model_path)
        print "\nTraining on " + split + ":"
        print "======================================\n"
        split_path  = os.path.join(pra_results_path,  split)
        output_path = os.path.join(expl_results_path, split)
        ensure_dir(output_path)

        target_relations = get_dirs(split_path) # get a list of target relations
        results = []

        for target_relation in target_relations:
            print("\tProcessing `{}`... ".format(target_relation)),

            if expl.load_data(split_path, target_relation):
                print("Loaded... "),

                # global logit
                expl.train_global_logit()
                expl.explain_model(output_path=output_path)
                # expl.explain_per_example(output_path)
                results.append(calc_metrics(expl))

                # global regression
                # expl.train_global_regression()
                # expl.explain_model(output_path=output_path)
                # expl.explain_per_example(output_path)
                # results.append(calc_metrics(expl))

                # local models
                # local logit
                # expl.train_local_logit_for_all(output_path, lfs.get_local_train_data)

                # local regression
                # expl.train_local_regression_for_all(output_path, lfs.get_local_train_data)
                print("Processed.")
            else:
                print("\tCould not load data for `{}`. Skipping relation...".format(target_relation))

        # save overall results
        pd.DataFrame(results).to_csv(output_path + '/overall_results.tsv', sep='\t')
    print("Pipeline finished.")



def process_overall_metrics(emb_import_paths, splits=None):
    metrics_dicts = []
    for emb_model_path in emb_import_paths:
        model_info = train_test.read_model_info(emb_model_path)
        pra_results_path  = emb_model_path + '/pra_explain/results/'
        expl_results_path = emb_model_path + '/pra_explain/results_explained/'
        if splits == None:
            splits = get_dirs(pra_results_path)
        for split in splits:
            output_path = os.path.join(expl_results_path, split)
            metrics_dicts.append(calc_overall_metrics(output_path + '/overall_results.tsv', model_info, split))
    return metrics_dicts

def process_overall_metrics_wo_emb(dataset_paths, splits=None):
    """Same as above but for when SFE was run directly on the dataset (not
    explaining an embedding model).
    """
    metrics_dicts = []
    for dataset_path in dataset_paths:
        # we have to make up model_info ourselves
        model_info = {
            'dataset_name': dataset_path.strip('/').split('/')[-1],
            'model_name': '-',
            'timestamp': '-',
        }
        pra_results_path  = dataset_path + '/pra/results/'
        expl_results_path = dataset_path + '/pra/results_explained/'
        if splits == None:
            splits = get_dirs(pra_results_path)
        for split in splits:
            output_path = os.path.join(expl_results_path, split)
            metrics_dicts.append(calc_overall_metrics(output_path + '/overall_results.tsv', model_info, split))
    return metrics_dicts
