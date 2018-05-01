import pandas as pd


def get_all_features(mtrx_df):
    """From an initial dataframe read from a `.tsv` feature matrix file outputted from
    the pra algorithm, returns a set that contains all features.
    """
    all_features = set()
    for i, row in mtrx_df.iterrows():
        for feat in row.features.split(' -#- '):
            feat_name, _ = feat.split(',')
            all_features.add(feat_name)
    return all_features


def parse_matrix_df(mtrx_df, all_features):
    """Parses an initial dataframe, read from a `.tsv` feature matrix file outputted from
    the pra algorithm, into another dataframe containing the following columns:

    - `label`: 1 for positive triples and -1 for negative ones
    - `head`: the head entity for the triple represented in the row
    - `tail`: same as above, but for the tail entity
    - `features`: one column for each feature with its respective value.
      If the feature was not observed for the triple, it will default to 0.
    """
    list_ = []
    for i, row in mtrx_df.iterrows():
        d = {}
        # initialize the features in 0 for each triple
        for feat_name in all_features:
            d[feat_name] = 0
        # get entities and label
        d['head'], d['tail'] = row.entities.split(',')
        d['label'] = row.label
        # update value for the features observed
        for feat in row.features.split(' -#- '):
            feat_name, value = feat.split(',')
            if feat_name in all_features:
                d[feat_name] = value
        list_.append(d)
    return pd.DataFrame(list_)


def parse_feature_matrices(train_fpath, test_fpath):
    """Parses training and test feature matrix files outputted from the pra algorithm.
    Returns a pandas dataframe for each file with the following columns according to the
    functions called in the code (see the other functions description).
    """
    # read files
    train_mtrx = pd.read_csv(train_fpath,
                             sep='\t',
                             names=['entities', 'label', 'features'])
    test_mtrx = pd.read_csv(test_fpath,
                            sep='\t',
                            names=['entities', 'label', 'features'])

    # build a set that contains all features (what matters are the training features only)
    all_features = get_all_features(train_mtrx)

    return (parse_matrix_df(train_mtrx, all_features),
            parse_matrix_df(test_mtrx, all_features))
