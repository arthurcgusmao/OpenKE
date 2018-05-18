"""Functions that get indices of train examples in the locality of (head, tail).

The idea is that we experiment with many different ways of getting examples in the locality of
(head, tail), and keeping the different functions in a separate file keeps the code cleaner and
easier to understand.
"""

def get_local_train_data(self, head, tail, y_mode="labels"):
    """Returns a dict containing the features and the labels/scores for the nearby examples, for
    training a local approximation.

    Arguments:
    - `self`: (Explanator) an instance of the Explanator class
    - `head`: (string) the head entity name
    - `tail`: (string) the tail entity name
    - `y_mode`: (string) "labels" or "scores", the kind of y data that should be returned
    """
    # get the nearest neighbors
    _, head_indices = self.get_kneighbors(head)
    _, tail_indices = self.get_kneighbors(tail)

    # get the corresponding training examples
    examples_indices = []
    for head_index in head_indices[0][1:]:
        examples_indices.extend(self.train_data.index[self.train_data['head'] == self.id2entity[head_index]].tolist())
    for tail_index in tail_indices[0][1:]:
        examples_indices.extend(self.train_data.index[self.train_data['tail'] == self.id2entity[tail_index]].tolist())
    self.n_nearby_examples = len(examples_indices)

    # get features
    train_x_local = self.train_x[examples_indices, :]

    # get labels or scores
    if y_mode == 'labels':
        train_y_local = self.train_y[examples_indices, :]
    if y_mode == 'scores':
        train_y_local = self.emb_predict(
                self.train_heads[examples_indices],
                self.train_tails[examples_indices],
                [self.target_relation] * len(examples_indices))

    return {
        'x': train_x_local,
        'y': train_y_local,
    }
