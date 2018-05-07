"""The goal of this file is to provide functions that help processing the
datasets."""

import pandas as pd
import numpy as np
import random


def create_name2id_dicts_from_df(df):
    """Creates two dictionaries from a dataframe containing two columns: one
    for the names (e.g., entities or relations) and another for the ids. The
    columns should be named 'name' and 'id', respectively.

    Arguments:
    - fpath: a pandas dataframe containing the data

    Returns:
    Two dictionaries: one that maps names to ids and another that maps ids to
    names
    """
    id2name = {row['id']: row['name'] for _, row in df.iterrows()}
    name2id = {name: id_ for id_, name in id2name.iteritems()}
    return name2id, id2name


def read_name2id_file(fpath, sep='\t', skiprows=1, id_first=False):
    """Reads a file that maps names (of entities or relations) to ids
    (numbers).

    Returns:
    Two dictionaries: one that maps names to ids and another that maps ids to
    names"""
    if id_first:
        names = ['id', 'name']
    else:
        names = ['name', 'id']
    df = pd.read_csv(fpath, sep=sep, skiprows=skiprows, names=names)
    return create_name2id_dicts_from_df(df)


def generate_name2id_files(dataset_dirpath, names_fname=['train.txt', 'test.txt', 'valid.txt'],
                           order=['head', 'relation', 'tail'], sep='\t', skiprows=0,
                           labels=[]):
    """Generates two files that maps names (of entities and relations) to ids
    (i.e., it encodes each name for entities and relations observed in the input
    files into numbers (ids)).

    Lines of the output files will be written in the following order:
        head entity, tail entity, relation

    Arguments:
    - dataset_dirpath: path to the dataset directory where all files should be in.
    - names_fpath: should be a list of file names (e.g., train.txt, test.txt, valid.txt).
    - labels: indicates which files have labels in the last row, so the function know what is the
    correct thing to do.
    """
    data = pd.DataFrame()
    for fname in names_fname:
        if fname in labels:
            _order = order + ['label']
        else:
            _order = order
        df = pd.read_csv(dataset_dirpath + fname, sep=sep, skiprows=skiprows, names=_order)
        if fname in labels:
            del df['label']
        data = pd.concat([data, df])

    relations = data['relation'].unique()
    entities = pd.concat([data['head'], data['tail']]).unique()

    with open(dataset_dirpath + 'entity2id.txt', 'w') as f:
        f.write("{}\n".format(len(entities)))
        for id, name in enumerate(entities):
            f.write("{}\t{}\n".format(name, id))
    with open(dataset_dirpath + 'relation2id.txt', 'w') as f:
        f.write("{}\n".format(len(relations)))
        for id, name in enumerate(relations):
            f.write("{}\t{}\n".format(name, id))


def generate_fold2id_files(dataset_dirpath, folds_fname=['train.txt', 'test.txt', 'valid.txt'],
                           extension='.txt', order=['head', 'relation', 'tail'], sep='\t',
                           skiprows=0, labels=[]):
    """Generates one fold2id file for each inputted fold file (e.g., for train.txt it will
    generate a train2id.txt file). It presupposes that there will be already two files in the
    dataset that map names to ids: `entity2id.txt` and `relation2id.txt`. The fold2id files
    generated will have triples written in the order (head, tail, relation).

    Arguments:
    - dataset_dirpath: path to the dataset directory where all files should be in.
    - folds_fpath: should be a list of fold file names (e.g., train.txt, test.txt, valid.txt).
    - order: the order that the triple is written in each line of the fold files. They should always
    be head, tail and relation, only the order that should change.
    - labels: indicates which files have labels in the last row, and that the function will
    generate two output files for each of those files (e.g., test2id.txt and test2id_neg.txt),
    one for positive and another for negative examples.
    """
    entity2id, _ = read_name2id_file(dataset_dirpath + 'entity2id.txt')
    relation2id, _ = read_name2id_file(dataset_dirpath + 'relation2id.txt')

    for fname in folds_fname:
        fold2id_fname = fname.replace(extension, '2id' + extension) # get the output filename
        neg_fold2id_fname = fname.replace(extension, '2id_neg' + extension) # get the negative output filename
        # check the existence of labels
        if fname in labels:
            _order = order + ['label']
        else:
            _order = order
        df = pd.read_csv(dataset_dirpath + fname, sep=sep, skiprows=skiprows, names=_order)
        df['head'] = df['head'].map(entity2id)
        df['tail'] = df['tail'].map(entity2id)
        df['relation'] = df['relation'].map(relation2id)

        if fname in labels: # for the case with labels
            # separate the dataframes into positive and negative if labels exist
            pos_df = df.loc[df['label'] == 1]
            neg_df = df.loc[df['label'] == -1]
            del pos_df['label']
            del neg_df['label']
            # write to csv files
            pos_df.to_csv(dataset_dirpath + fold2id_fname, columns=['head', 'tail', 'relation'],
                          index=False, header=False, sep=' ')
            neg_df.to_csv(dataset_dirpath + neg_fold2id_fname, columns=['head', 'tail', 'relation'],
                          index=False, header=False, sep=' ')
            # insert row number in the beginning of files
            with open(dataset_dirpath + fold2id_fname, 'r') as f: content = f.read()
            with open(dataset_dirpath + fold2id_fname, 'w') as f: f.write("{}\n".format(len(pos_df)) + content)
            with open(dataset_dirpath + neg_fold2id_fname, 'r') as f: content = f.read()
            with open(dataset_dirpath + neg_fold2id_fname, 'w') as f: f.write("{}\n".format(len(neg_df)) + content)

        else: # for the case without labels
            df.to_csv(dataset_dirpath + fold2id_fname, columns=['head', 'tail', 'relation'],
                      index=False, header=False, sep=' ')
            # insert row number in the beginning of file
            with open(dataset_dirpath + fold2id_fname, 'r') as f: content = f.read()
            with open(dataset_dirpath + fold2id_fname, 'w') as f: f.write("{}\n".format(len(df)) + content)



def generate_type_constrain_file(dataset_path='./', fname='/type_constrain.txt'):
    """Creates a new file named `type_constrain.txt` that contains two lines for
    each relation in the dataset. The first line contains information on the
    entities that were observed as head (first position) of the relation. The
    second one contains information on the entities that were observed as being
    in the tail (last position) of the relation. Each line contains the
    following columns:

    - Column 1: the relation
    - Column 2: the number of different entities that were observed in that
      position (head or tail)
    - Columns 3 onwards: each different entity observed in the respective
      position

    The first line of the file will contain the number of relations observed in
    the dataset.

    Arguments:
    - dataset_path: the path to where the dataset is located.
    - fname: the name of the file to be generated.
    """
    train_fpath = dataset_path + '/train2id.txt'
    valid_fpath = dataset_path + '/valid2id.txt'
    test_fpath = dataset_path + '/test2id.txt'
    sep = ' '
    names = ['e1', 'e2', 'rel']

    train = pd.read_csv(train_fpath, sep=sep, skiprows=1, names=names)
    valid = pd.read_csv(valid_fpath, sep=sep, skiprows=1, names=names)
    test = pd.read_csv(test_fpath, sep=sep, skiprows=1, names=names)

    data = pd.concat([train, valid, test])

    # create a list containing the information
    rels = data.rel.unique() # get all relations contained in the dataframe
    out_list = []
    for rel in rels:
        rel_rows = data.loc[data.rel == rel] # get only rows with the respective relation
        e1s = rel_rows.e1.unique()
        e2s = rel_rows.e2.unique()
        out_list.append(np.concatenate(([rel, len(e1s)], e1s)))
        out_list.append(np.concatenate(([rel, len(e2s)], e2s)))

    # write information to file
    with open(dataset_path + fname, 'w') as f:
        f.write(str(len(rels)))
        f.write('\n')
        for line in out_list:
            line = [str(i) for i in line] # convert everything to string
            f.write('\t'.join(line)) # tab-separate everything
            f.write('\n')


def read_type_constrain_file(filepath):
    """Parses a type_constrain file into a dict. Each key in the dict is a relation and each value
    another dict. This second dict has two keys: head and tail. Each value of this second dict is a
    set of entities that were observed in that position (head or tail) of the graph for the
    respective relation.
    """
    with open(filepath, 'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        del content[0] # remove the first line
        content = [line.split('\t') for line in content]
        content = [[int(item) for item in line] for line in content]
        output = {}
        last_rel = None
        for line in content:
            rel = line[0]
            entities = line[2:]
            if not rel == last_rel:
                output[rel] = {'head': set(entities)}
            else:
                output[rel]['tail'] = set(entities)
            last_rel = rel
    return output


def ensure_one_to_one_negative_examples(dataset_path):
    """The goal of this function is to ensure that for each positive observation
    in the dataset there was generated exactly one negative observation by
    corrupting the triple. It also ensures that the observations follow the same
    order.

    It presupposes that there are the `test2id.txt` and `test2id_neg.txt` (or valid) files.
    """
    test_fpath = dataset_path + '/test2id.txt'
    valid_fpath = dataset_path + '/valid2id.txt'
    test_neg_fpath = dataset_path + '/test2id_neg.txt'
    valid_neg_fpath = dataset_path + '/valid2id_neg.txt'
    sep = ' '
    names = ['e1', 'e2', 'rel']

    valid = pd.read_csv(valid_fpath, sep=sep, skiprows=1, names=names)
    test = pd.read_csv(test_fpath, sep=sep, skiprows=1, names=names)
    valid_neg = pd.read_csv(valid_neg_fpath, sep=sep, skiprows=1, names=names)
    test_neg = pd.read_csv(test_neg_fpath, sep=sep, skiprows=1, names=names)

    flag = False
    for pos, neg in [(valid, valid_neg), (test, test_neg)]:
        for index, row in pos.iterrows():
            if row.rel != neg.iloc[index].rel:
                flag = True
                raise Exception('There is NOT a one-to-one relation between positive and negative examples.')
    if not flag:
        print('There seems to be everything ok with the proportion of negative and positive examples.')
    else:
        print('There is NOT a one-to-one relation between positive and negative examples.')


def create_inputs_for_openke(dataset_path):
    """This function calls other functions defined above in order to setup all input files needed
    for OpenKE. It assumes that in `dataset_path` there are already three files (train.txt,
    valid.txt, and test.txt) in the order (head, relation, tail) and that valid and test already
    contain negative examples (labels).
    """
    generate_name2id_files(dataset_path, labels=['valid.txt', 'test.txt'])
    generate_fold2id_files(dataset_path, labels=['valid.txt', 'test.txt'])
    generate_type_constrain_file(dataset_path=dataset_path)
    ensure_one_to_one_negative_examples(dataset_path)


def get_bern_prob_corrupt_tail(type_constrain_dict):
    """Gets the probability of the tail be corrupted when generating negative examples using the
    Bernoulli distribution proposed by Wang et al. (2014).

    Arguments:
    - type_constrain_dict: A dictionary where each key in the dict is a relation and each value
    another dict. This second dict has two keys: head and tail. Each value of this second dict is a
    set of entities that were observed in that position (head or tail) of the graph for the
    respective relation. See `read_type_constrain_file()` function.
    """
    relations = range(len(type_constrain_dict))
    prob_corrupt_tail = {}
    for r in relations:
        tph = float(len(type_constrain_dict[r]['tail'])) / len(type_constrain_dict[r]['head'])
        hpt = tph**(-1) # head_per_tail is the inverse of tail_per_head
        prob_corrupt_tail[r] = hpt / (hpt + tph)
    return prob_corrupt_tail


def generate_corrupted_training_examples(dataset_path, neg_proportion=1, bern=True,
                                         output_include_pos=True):
    """Generates negative examples for training following the Bernoulli sampling procedure proposed
    by Wang el al. (2014) if `bern=True` or using a uniform distribution if `bern=False`. A list is
    returned, where each element is a dict representing a triple (with keys = head, tail and
    relation).

    Arguments:
    - dataset_path: path of the dataset for which positive training examples will be corrupted
    - neg_proportion: proportion of negative examples for each positive one
    - bern: flag indicating that the bernoulli distribution will be used to corrupt head vs tail
    """
    train_triples = pd.read_csv(dataset_path + '/train2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    tc_dict = read_type_constrain_file(dataset_path + '/type_constrain.txt')
    # get a set of all entities present in test data
    ents = set()
    ents.update(train_triples['head'].unique())
    ents.update(train_triples['tail'].unique())

    if bern:
        prob_corrupt_tail = get_bern_prob_corrupt_tail(tc_dict)
    else:
        relations = train_triples['relation'].unique()
        prob_corrupt_tail = {rel: 0.5 for rel in relations}

    output = []
    for idx, row in train_triples.iterrows():
        r = row['relation']
        h = row['head']
        t = row['tail']
        if output_include_pos:
            output.append({'head': h,
                           'tail': t,
                           'relation': r,
                           'label': 1})
        for _ in range(neg_proportion):
            if prob_corrupt_tail[r] < random.random():
                t_ = random.sample(ents, 1)[0]
                h_ = h
            else:
                h_ = random.sample(ents, 1)[0]
                t_ = t
            output.append({'head': h_,
                           'tail': t_,
                           'relation': r,
                           'label': -1})
    return output
