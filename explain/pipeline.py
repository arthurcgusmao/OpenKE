from helpers import ensure_dir, get_dirs
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
                # expl.explain_per_example(data_path, 'test')
                results.append(expl.get_results())

                # global regression
                expl.train_global_regression()
                expl.explain_model(output_path=output_path)
                # expl.explain_per_example(data_path, 'test')
                results.append(expl.get_results())

                # local logit
                part_results = []
                for datapoint in test_set:
                    expl.train_local_logit("francis_ii", "English", lfs.get_local_train_data)
                ????????????
                # expl.explain_per_example(data_path, 'test')

                # local regression
                expl.train_local_regression()
                expl.explain_model(output_path=output_path)
                # expl.explain_per_example(data_path, 'test')
                results.append(expl.get_results())
            else:
                print("Could not load data for `{}`. Skipping relation...".format(target_relation))

        # save overall results
        pd.DataFrame(results).to_csv(output_path + '/overall_results.tsv', sep='\t')
