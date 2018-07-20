import json
import os

import numpy as np


class RNG(np.random.RandomState):
    def __init__(self, seed=None):
        self._seed = seed
        super(RNG, self).__init__(self._seed)

    def reseed(self):
        if self._seed is not None:
            self.seed(self._seed)


class TrainTestSplitter(object):
    def __init__(self, shuffle=False, random_seed=None):
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.rng = RNG(self.random_seed)

    def split(self, y, train_ratio=0.8, stratify=False):
        self.rng.reseed()
        n = len(y)

        if not stratify:
            indices = (
                self.rng.permutation(n) if self.shuffle else np.arange(n, dtype=np.int)
            )
            train_size = int(train_ratio * n)
            return np.split(indices, (train_size,))

        # group indices by label
        labels_indices = {}
        for index, label in enumerate(y):
            if not label in labels_indices:
                labels_indices[label] = []
            labels_indices[label].append(index)

        train, test = np.array([], dtype=np.int), np.array([], dtype=np.int)
        for label, indices in sorted(labels_indices.items()):
            size = int(train_ratio * len(indices))
            train = np.concatenate((train, indices[:size]))
            test = np.concatenate((test, indices[size:]))

        if self.shuffle:
            self.rng.shuffle(train)
            self.rng.shuffle(test)

        return train, test

    def make_k_folds(self, y, n_folds=3, stratify=False):
        """
        Split data into folds of (approximately) equal size.

        Parameters
        ----------
        y : (n_samples,) array-like
            The target variable for supervised learning problems.
            Stratification is done based upon the `y` labels.
        n_folds : int, `n_folds` > 1, optional
            Number of folds.
        stratify : bool, optional
            If True, the folds are made by preserving the percentage of samples
            for each class. Stratification is done based upon the `y` labels.

        Yields
        ------
        fold : np.ndarray
            Indices for current fold.
        """
        self.rng.reseed()
        n = len(y)

        if not stratify:
            indices = (
                self.rng.permutation(n) if self.shuffle else np.arange(n, dtype=np.int)
            )
            for fold in np.array_split(indices, n_folds):
                yield fold
            return

        # group indices
        labels_indices = {}
        for index, label in enumerate(y):
            if isinstance(label, np.ndarray):
                label = tuple(label.tolist())
            if not label in labels_indices:
                labels_indices[label] = []
            labels_indices[label].append(index)

        # split all indices label-wisely
        for label, indices in sorted(labels_indices.items()):
            labels_indices[label] = np.array_split(indices, n_folds)

        # collect respective splits into folds and shuffle if needed
        for k in range(n_folds):
            fold = np.concatenate(
                [indices[k] for _, indices in sorted(labels_indices.items())]
            )
            if self.shuffle:
                self.rng.shuffle(fold)
            yield fold

    def k_fold_split(self, y, n_splits=3, stratify=False):
        """
        Split data into train and test subsets for K-fold CV.

        Parameters
        ----------
        y : (n_samples,) array-like
            The target variable for supervised learning problems.
            Stratification is done based upon the `y` labels.
        n_splits : int, `n_splits` > 1, optional
            Number of folds.
        stratify : bool, optional
            If True, the folds are made by preserving the percentage of samples
            for each class. Stratification is done based upon the `y` labels.

        Yields
        ------
        train : (n_train,) np.ndarray
            The training set indices for current split.
        test : (n_samples - n_train,) np.ndarray
            The testing set indices for current split.
        """
        folds = list(self.make_k_folds(y, n_folds=n_splits, stratify=stratify))
        for i in range(n_splits):
            yield np.concatenate(folds[:i] + folds[(i + 1) :]), folds[i]


def one_hot(y):
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]


def get_initialization(initialization_name):
    for k, v in globals().items():
        if k.lower() == initialization_name.lower():
            return v
    raise ValueError("invalid initialization name '{0}'".format(initialization_name))


def get_activation(activation_name):
    for k, v in globals().items():
        if k.lower() == activation_name.lower():
            return v
    raise ValueError("invalid activation function name '{0}'".format(activation_name))


def pformat(params, offset, printer=repr):
    np_print_options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=32, edgeitems=2)

    params_strs = []
    current_line_len = offset
    line_sep = ",\n" + min(1 + offset / 2, 8) * " "

    for key, value in sorted(params.items()):
        this_repr = "{0}={1}".format(key, printer(value))
        if len(this_repr) > 256:
            this_repr = this_repr[:192] + "..." + this_repr[-64:]
        if current_line_len + len(this_repr) >= 75 or "\n" in this_repr:
            params_strs.append(line_sep)
            current_line_len = len(line_sep)
        elif params_strs:
            params_strs.append(", ")
            current_line_len += 2
        params_strs.append(this_repr)
        current_line_len += len(this_repr)

    np.set_printoptions(**np_print_options)

    pformatted = "".join(params_strs)
    # strip trailing space to avoid nightmare in doctests
    pformatted = "\n".join(l.rstrip() for l in pformatted.split("\n"))
    return pformatted


def get_metric(metric_name):
    for k, v in globals().items():
        if k.lower() == metric_name.lower():
            return v
    raise ValueError("invalid metric name '{0}'".format(metric_name))


def get_optimizer(optimizer_name, **params):
    for k, v in globals().items():
        if k.lower() == optimizer_name.lower():
            return v(**params)
    raise ValueError("invalid optimizer name '{0}'".format(optimizer_name))


def save_model(model, filepath=None, params_mask={}, json_params={}):
    filepath = filepath or "model.json"
    params = model.get_params(deep=False, **params_mask)
    params = model._serialize(params)
    with open(filepath, "w") as f:
        json.dump(params, f, **json_params)


def one_hot_decision_function(y):
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z


def import_trace(
    module_path,
    main_package_name,
    include_main_package=True,
    discard_underscore_packages=True,
):
    trace = ""
    head = module_path
    while True:
        head, tail = os.path.split(head)
        tail = os.path.splitext(tail)[0]
        if discard_underscore_packages and tail.startswith("_"):
            continue
        if not tail:
            raise ValueError(
                "main package name '{0}' is not a part of '{1}'".format(
                    main_package_name, module_path
                )
            )
        if tail == main_package_name:
            if include_main_package:
                trace = ".".join(filter(bool, [tail, trace]))
            return trace
        trace = ".".join(filter(bool, [tail, trace]))
    return trace


def is_param_name(name):
    return not name.startswith("_") and not name.endswith("_")


def is_param_or_attribute_name(name):
    return not name.startswith("_")
