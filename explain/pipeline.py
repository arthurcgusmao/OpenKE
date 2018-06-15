import os
import pandas as pd

from helpers import ensure_dir, get_dirs, get_metrics
from Explanator import Explanator
import local_functions as lfs


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
        print "\nTraining on " + split
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
                results.append(expl.get_results())

                # global regression
                # expl.train_global_regression()
                # expl.explain_model(output_path=output_path)
                # expl.explain_per_example(output_path)
                # results.append(expl.get_results())

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
        if 'metrics_frame' in globals():
            global metrics_frame
            metrics_frame = metrics_frame.append(get_metrics(output_path + '/overall_results.tsv', split), ignore_index=True)


if __name__ == "__main__":
    paths = ["/Users/Alvinho/openke/results/FB13/TransE/1527033688",
             "/Users/Alvinho/openke/results/NELL186/TransE/1526711822",
             "/Users/Alvinho/openke/results/WN11/TransE/1527008113"]

    global metrics_frame
    metrics_frame = pd.DataFrame()
    for path in paths:
        pipeline(path)
        metrics_frame.to_csv('/Users/Alvinho/openke/results/global_metrics.tsv', sep='\t', index=False)
