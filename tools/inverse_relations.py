"""The idea here is to extend existing datasets by adding the reverse of existing relations (in the
case the relation does not already have its inverse) and see if it improves the accuracy.
"""

import pandas as pd
import dataset_tools


def inverse_data(fold):
    fold_inversed = fold.copy()
    fold_inversed['tail'] = fold['head']
    fold_inversed['head'] = fold['tail']
    fold_inversed['relation'] = fold['relation'].map(lambda x: '_' + x)
    return fold_inversed


def inverse_dataset(dataset_path):
    """Extends a dataset by adding for each relation an inverse relation. It does not check for the
    existence of inverse relations, it will duplicate each relation mechanically.

    This funciton
    assumes that in `dataset_path` there are already three files (train.txt, valid.txt, and
    test.txt) in the order (head, relation, tail) and that valid and test already contain negative
    examples (labels).
    """
    train = pd.read_csv(dataset_path + 'train.txt', skiprows=0, sep='\t', names=['head', 'relation', 'tail'])
    valid = pd.read_csv(dataset_path + 'valid.txt', skiprows=0, sep='\t', names=['head', 'relation', 'tail', 'label'])
    test = pd.read_csv(dataset_path + 'test.txt', skiprows=0, sep='\t', names=['head', 'relation', 'tail', 'label'])

    train_inv = inverse_data(train)
    valid_inv = inverse_data(valid)
    test_inv = inverse_data(test)

    train = pd.concat((train, train_inv))
    valid = pd.concat((valid, valid_inv))
    test = pd.concat((test, test_inv))

    train.to_csv(dataset_path + 'train.txt', sep='\t', index=False, header=False, columns=['head', 'relation', 'tail'])
    valid.to_csv(dataset_path + 'valid.txt', sep='\t', index=False, header=False, columns=['head', 'relation', 'tail', 'label'])
    test.to_csv(dataset_path + 'test.txt', sep='\t', index=False, header=False, columns=['head', 'relation', 'tail', 'label'])

    dataset_tools.create_inputs_for_openke(dataset_path)
