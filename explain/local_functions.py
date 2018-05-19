"""Functions that get indices of train examples in the locality of (head, tail).

The idea is that we experiment with many different ways of getting examples in the locality of
(head, tail), and keeping the different functions in a separate file keeps the code cleaner and
easier to understand.
"""

import numpy as np


def get_local_train_data(self, head, tail, y_type="labels"):
    """Returns a dict containing the features and the labels/scores for the nearby examples, for
    training a local approximation.

    Arguments:
    - `self`: (Explanator) an instance of the Explanator class
    - `head`: (string) the head entity name
    - `tail`: (string) the tail entity name
    - `y_type`: (string) "labels" or "scores", the kind of y data that should be returned
    """
    # get the nearest neighbors
    _, head_indices = self.get_kneighbors(head)
    _, tail_indices = self.get_kneighbors(tail)

    # Get index for the target relation
    head_ix = np.where(self.test_heads == head)[0]
    tail_ix = np.where(self.test_tails == tail)[0]
    index = list(set(head_ix).intersection(tail_ix))

    # get the corresponding training examples
    examples_indices = []
    for head_id in head_indices[0][1:]:
        examples_indices.extend(np.where(self.train_heads == self.id2entity[head_id])[0])
    for tail_id in tail_indices[0][1:]:
        examples_indices.extend(np.where(self.train_tails == self.id2entity[tail_id])[0])
    self.n_nearby_examples = len(examples_indices)

    # get features
    train_x_local = self.train_x[examples_indices, :]

    # get labels or scores
    if y_type == 'labels':
        train_y_local = self.train_y[examples_indices, :]
    if y_type == 'scores':
        train_y_local = self.emb_predict(
                self.train_heads[examples_indices],
                self.train_tails[examples_indices],
                [self.target_relation] * len(examples_indices))

    return {
        'x': train_x_local,
        'y': train_y_local,
        'index': index
    }
