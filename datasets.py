from collections import OrderedDict
from itertools import islice
from os import path

import numpy as np
import pandas as pd
from scipy.io import mmread
from torchvision.transforms import Compose
import torch

from pyll.base import TorchDataset
from pyll.utils.misc import invoke_functional_with_params
from pyll.utils.timer import Timer


class Cellpainting(TorchDataset):
    def __init__(self, sample_index_file: str, data_directory_path: str, label_matrix_file: str = None,
                 label_row_index_file: str = None, label_col_index_file: str = None, auxiliary_labels=None,
                 transforms=None, group_views: bool = False,
                 subset: float = 1., num_classes: int = None, verbose: bool = False):
        """ Read samples from cellpainting dataset."""
        self.verbose = verbose
        assert (path.exists(sample_index_file))
        assert (path.exists(data_directory_path))

        # Read sample index
        sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
        sample_index.set_index(["SAMPLE_KEY"])

        # read auxiliary labels if provided
        if auxiliary_labels is not None:
            pddata = pd.read_csv(auxiliary_labels, sep=",", header=0)
            self.auxiliary_data = pddata.as_matrix()[:, 2:].astype(np.float32)
            # threshold
            self.auxiliary_data[self.auxiliary_data < 0.75] = -1
            self.auxiliary_data[self.auxiliary_data >= 0.75] = 1
            self.auxiliary_assays = list(pddata)[2:]
            self.n_auxiliary_classes = len(self.auxiliary_assays)
            self.auxiliary_smiles = pddata["SMILES"].tolist()
        else:
            self.n_auxiliary_classes = 0

        if group_views:
            sample_groups = sample_index.groupby(['PLATE_ID', 'WELL_POSITION'])
            sample_keys = list(sample_groups.groups.keys())
            sample_index = sample_groups
            self.sample_to_smiles = None  # TODO
        else:
            sample_keys = sample_index['SAMPLE_KEY'].tolist()
            if auxiliary_labels is not None:
                self.sample_to_smiles = dict(zip(sample_index.SAMPLE_KEY, [self.auxiliary_smiles.index(s) for s in sample_index.SMILES]))
            else:
                self.sample_to_smiles = None

        if len(sample_keys) == 0:
            raise Exception("Empty dataset!")
        else:
            self.log("Found {} samples".format(len(sample_keys)))

        if subset != 1.:
            sample_keys = sample_keys[:int(len(sample_keys) * subset)]

        # Read Label Matrix if specified
        if label_matrix_file is not None:
            assert (path.exists(label_matrix_file))
            assert (path.exists(label_row_index_file))
            assert (path.exists(label_col_index_file))

            if label_row_index_file is not None and label_col_index_file is not None:
                col_index = pd.read_csv(label_col_index_file, sep=",", header=0)
                row_index = pd.read_csv(label_row_index_file, sep=",", header=0)
                label_matrix = mmread(label_matrix_file).tocsr()
                # --
                self.label_matrix = label_matrix
                self.row_index = row_index
                self.col_index = col_index
                if group_views:
                    self.label_dict = dict(
                        (key, sample_groups.get_group(key).iloc[0].ROW_NR_LABEL_MAT) for key in sample_keys)
                else:
                    self.label_dict = dict(zip(sample_index.SAMPLE_KEY, sample_index.ROW_NR_LABEL_MAT))
                self.n_classes = label_matrix.shape[1]
            else:
                raise Exception("If label is specified index files must be passed!")
        else:
            self.label_matrix = None
            self.row_index = None
            self.col_index = None
            self.label_dict = None
            self.n_classes = num_classes

        if auxiliary_labels is not None:
            self.n_classes += self.n_auxiliary_classes

        # expose everything important
        self.data_directory = data_directory_path
        self.sample_index = sample_index
        self.n_samples = len(sample_keys)
        self.sample_keys = sample_keys
        self.group_views = group_views
        self.transforms = transforms

        # load first sample and check shape
        i = 0
        sample = self[i]

        while sample["input"] is None and i < len(self):
            sample = self[i]
            i += 1

        if sample["input"] is not None:
            self.data_shape = sample["input"].shape
        else:
            self.data_shape = "Unknown"
        self.log("Discovered {} samples (subset={}) with shape {}".format(self.n_samples, subset, self.data_shape))

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        sample_key = self.sample_keys[idx]
        return self.read_sample(sample_key)

    @property
    def shape(self):
        return self.data_shape

    @property
    def num_classes(self):
        return self.n_classes

    def log(self, message):
        if self.verbose:
            print(message)

    def read_sample(self, key):
        with Timer("Read Sample", verbose=self.verbose):
            if self.group_views:
                X = self.load_view_group(key)
            else:
                filepath = path.join(self.data_directory, "{}.npz".format(key))
                if path.exists(filepath):
                    X = self.load_view(filepath=filepath)
                else:
                    print("ERROR: Missing sample '{}'".format(key))
                    return dict(input=None, ID=key)

            if self.transforms:
                X = self.transforms(X)

            # get label
            if self.label_dict is not None:
                label_idx = self.label_dict[key]
                y = self.label_matrix[label_idx].toarray()[0].astype(np.float32)
                if self.sample_to_smiles is not None and key in self.sample_to_smiles:
                    y = np.concatenate([y, self.auxiliary_data[self.sample_to_smiles[key], :]])

                return dict(input=X, target=y, ID=key)
            else:
                return dict(input=X, ID=key)

    def get_sample_keys(self):
        return self.sample_keys.copy()

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath)
        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            # for c in range(image.shape[-1]):
                # image[:, :, c] = (image[:, :, c] - image[:, :, c].mean()) / image[:, :, c].std()
                # image[:, :, c] = ((image[:, :, c] - image[:, :, c].mean()) / image[:, :, c].std() * 255).astype(np.uint8)
            # image = (image - image.mean()) / image.std()
            return image

        return None

    def load_view_group(self, groupkey):
        result = np.empty((1040, 2088 - 12, 5), dtype=np.uint8)
        viewgroup = self.sample_index.get_group(groupkey)
        for i, view in enumerate(viewgroup.sort_values("SITE", ascending=True).iterrows()):
            corner = (0 if int(i / 3) == 0 else 520, i % 3 * 692)
            filepath = path.join(self.data_directory, "{}.npz".format(view[1].SAMPLE_KEY))
            v = self.load_view(filepath=filepath)[:, 4:, :]
            # for j in range(v.shape[-1]):
            #    plt.imshow(v[:, :, j])
            #    plt.savefig("{}-{}-{}-{}.png".format(groupkey[0], groupkey[1], i, j))
            result[corner[0]:corner[0] + 520, corner[1]:corner[1] + 692, :] = v
        return result


class CellpaintingSingleCell(TorchDataset):
    def __init__(self, sample_index_file: str, data_directory_path: str, label_matrix_file: str = None,
                 label_row_index_file: str = None, label_col_index_file: str = None,
                 transforms=None,  subset: float = 1., num_classes: int = None, verbose: bool = False):
        """ Read samples from cellpainting dataset."""
        self.verbose = verbose

        assert (path.exists(sample_index_file))
        assert (path.exists(data_directory_path))

        # Read sample index
        sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
        sample_index.set_index(["SAMPLE_KEY"])
        sample_keys = sample_index['SAMPLE_KEY'].tolist()

        if len(sample_keys) == 0:
            raise Exception("Empty dataset!")
        else:
            self.log("Found {} samples".format(len(sample_keys)))

        if subset != 1.:
            sample_keys = sample_keys[:int(len(sample_keys) * subset)]

        # Read Label Matrix if specified
        if label_matrix_file is not None:
            assert (path.exists(label_matrix_file))
            assert (path.exists(label_row_index_file))
            assert (path.exists(label_col_index_file))

            if label_row_index_file is not None and label_col_index_file is not None:
                col_index = pd.read_csv(label_col_index_file, sep=",", header=0)
                row_index = pd.read_csv(label_row_index_file, sep=",", header=0)
                label_matrix = mmread(label_matrix_file).tocsr()
                # --
                self.label_matrix = label_matrix
                self.row_index = row_index
                self.col_index = col_index
                self.label_dict = dict(zip(sample_index.SAMPLE_KEY, sample_index.ROW_NR_LABEL_MAT))
                self.n_classes = label_matrix.shape[1]
            else:
                raise Exception("If label is specified index files must be passed!")
        else:
            self.label_matrix = None
            self.row_index = None
            self.col_index = None
            self.label_dict = None
            self.n_classes = num_classes

        # expose everything important
        self.data_directory = data_directory_path
        self.sample_index = sample_index
        self.n_samples = len(sample_keys)
        self.sample_keys = sample_keys
        self.transforms = transforms

        # load first sample and check shape
        i = 0
        sample = self[i]

        while sample["input"] is None and i < len(self):
            sample = self[i]
            i += 1

        if sample["input"] is not None:
            self.data_shape = sample["input"].shape
        else:
            self.data_shape = "Unknown"
        self.log("Discovered {} samples (subset={}) with shape {}".format(self.n_samples, subset, self.data_shape))

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        sample_key = self.sample_keys[idx]
        return self.read_sample(sample_key)

    @property
    def shape(self):
        return self.data_shape

    @property
    def num_classes(self):
        return self.n_classes

    def log(self, message):
        if self.verbose:
            print(message)

    def read_sample(self, key):
        with Timer("Read Sample", verbose=self.verbose):
            filepath = path.join(self.data_directory, "{}.npz".format(key))
            if path.exists(filepath):
                X = self.load_view(filepath=filepath)
            else:
                print("ERROR: Missing sample '{}'".format(key))
                return dict(input=None, ID=key)

            if self.transforms:
                X = self.transforms(X)

            # get label
            if self.label_dict is not None:
                label_idx = self.label_dict[key]
                y = self.label_matrix[label_idx].toarray()[0].astype(np.float32)
                return dict(input=X, target=y, ID=key)
            else:
                return dict(input=X, ID=key)

    def get_sample_keys(self):
        return self.sample_keys.copy()

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath)
        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            # for c in range(image.shape[-1]):
                # image[:, :, c] = (image[:, :, c] - image[:, :, c].mean()) / image[:, :, c].std()
                # image[:, :, c] = ((image[:, :, c] - image[:, :, c].mean()) / image[:, :, c].std() * 255).astype(np.uint8)
            # image = (image - image.mean()) / image.std()
            return image

        return None

    def load_view_group(self, groupkey):
        result = np.empty((1040, 2088 - 12, 5), dtype=np.uint8)
        viewgroup = self.sample_index.get_group(groupkey)
        for i, view in enumerate(viewgroup.sort_values("SITE", ascending=True).iterrows()):
            corner = (0 if int(i / 3) == 0 else 520, i % 3 * 692)
            filepath = path.join(self.data_directory, "{}.npz".format(view[1].SAMPLE_KEY))
            v = self.load_view(filepath=filepath)[:, 4:, :]
            # for j in range(v.shape[-1]):
            #    plt.imshow(v[:, :, j])
            #    plt.savefig("{}-{}-{}-{}.png".format(groupkey[0], groupkey[1], i, j))
            result[corner[0]:corner[0] + 520, corner[1]:corner[1] + 692, :] = v
        return result


class CellpaintingPrecalculated(TorchDataset):
    def __init__(self, sample_index_file: str, data_file: str, label_matrix_file: str = None,
                 label_row_index_file: str = None, label_col_index_file: str = None, auxiliary_labels=None,
                 subset: float = 1., num_classes: int = None, verbose: bool = False):
        """ Read samples from cellpainting dataset."""
        self.verbose = verbose

        assert (path.exists(sample_index_file))
        assert (path.exists(data_file))

        # Read features from datafile
        with Timer("Reading feature matrix"):
            features = np.load(data_file)
            feature_rownames = {k: v for v, k in enumerate(list(features["rownames"]))}
            feature_matrix = features["X"]

        if auxiliary_labels is not None:
            with Timer("Reading aux labels"):
                pddata = pd.read_csv(auxiliary_labels, sep=",", header=0)
                self.auxiliary_data = pddata.as_matrix()[:, 2:].astype(np.float32)
                # threshold
                self.auxiliary_data[self.auxiliary_data < 0.5] = -1
                self.auxiliary_data[self.auxiliary_data >= 0.5] = 1
                self.auxiliary_assays = list(pddata)[2:]
                self.n_auxiliary_classes = len(self.auxiliary_assays)
                auxiliary_smiles = {k: v for v, k in enumerate(pddata["SMILES"].tolist())}
                # self.sample_to_smiles = dict(zip(sample_index.SAMPLE_KEY, [self.auxiliary_smiles.index(s) for s in sample_index.SMILES]))
        else:
            self.n_auxiliary_classes = 0
            auxiliary_smiles = None
            self.auxiliary_data = None
            # self.sample_to_smiles = None

        # Read sample index
        with Timer("Reading index file"):
            sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
            sample_groups = sample_index.groupby(['PLATE_ID', 'WELL_POSITION'])
            # sample_keys = list(sample_groups.groups.keys())
            # sample_index = sample_groups

            sample_keys = set([(p, w) for p, w in zip(sample_index.PLATE_ID, sample_index.WELL_POSITION)])
            sample_dict = OrderedDict()
            for key in sample_keys:
                group_sample = sample_groups.get_group(key).sort_values("SITE", ascending=True).iloc[0]
                try:
                    sample_dict[key] = {"smiles": group_sample.SMILES, "row_nr_label_mat": group_sample.ROW_NR_LABEL_MAT,
                                        "row_nr_feature_mat": feature_rownames["{}-{}".format(key[0], key[1])]}
                    if auxiliary_smiles is not None:
                        sample_dict[key]["aux_idx"] = auxiliary_smiles[group_sample.SMILES]
                except KeyError:
                    print("Missing {}".format("{}-{}".format(key[0], key[1])))

        # read auxiliary labels if provided

        if len(sample_dict) == 0:
            raise Exception("Empty dataset!")
        else:
            self.log("Found {} samples".format(len(sample_dict)))

        if subset != 1.:
            sample_dict = OrderedDict(islice(sample_dict.items(), int(len(sample_keys) * subset)))
            # sample_dict = sample_dict[:int(len(sample_keys) * subset)]

        # Read Label Matrix if specified
        if label_matrix_file is not None:
            assert (path.exists(label_matrix_file))
            assert (path.exists(label_row_index_file))
            assert (path.exists(label_col_index_file))

            if label_row_index_file is not None and label_col_index_file is not None:
                with Timer("Reading label matrix"):
                    col_index = pd.read_csv(label_col_index_file, sep=",", header=0)
                    row_index = pd.read_csv(label_row_index_file, sep=",", header=0)
                    label_matrix = mmread(label_matrix_file).tocsr()
                    # --
                    self.label_matrix = label_matrix
                    self.row_index = row_index
                    self.col_index = col_index
                    # self.label_dict = dict(zip(sample_index.SAMPLE_KEY, sample_index.ROW_NR_LABEL_MAT))
                    self.n_classes = label_matrix.shape[1]
            else:
                raise Exception("If label is specified index files must be passed!")
        else:
            self.label_matrix = None
            self.row_index = None
            self.col_index = None
            # self.label_dict = None
            self.n_classes = num_classes

        if auxiliary_labels is not None:
            self.n_classes += self.n_auxiliary_classes

        # expose everything important
        self.sample_dict = sample_dict
        self.sample_dict_idx = list(sample_dict.keys())
        self.feature_matrix = feature_matrix
        self.n_samples = len(sample_dict)

        # load first sample and check shape
        sample = self[0]
        if sample["input"] is not None:
            self.data_shape = sample["input"].shape
        else:
            self.data_shape = "Unknown"
        self.log("Discovered {} samples (subset={}) with shape {}".format(self.n_samples, subset, self.data_shape))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample_key = self.sample_dict_idx[idx]
        return self.read_sample(sample_key)

    @property
    def shape(self):
        return self.data_shape

    @property
    def num_classes(self):
        return self.n_classes

    def log(self, message):
        if self.verbose:
            print(message)

    def read_sample(self, key):
        with Timer("Read Sample", verbose=self.verbose):
            X = self.feature_matrix[self.sample_dict[key]["row_nr_feature_mat"]]
            X = (X - X.mean()) / X.std()
            # get label
            if self.label_matrix is not None:
                label_idx = self.sample_dict[key]["row_nr_label_mat"]
                y = self.label_matrix[label_idx].toarray()[0].astype(np.float32)
                if self.auxiliary_data is not None:
                    aux_idx = self.sample_dict[key]["aux_idx"]
                    y = np.concatenate([y, self.auxiliary_data[aux_idx, :]])

                return dict(input=X, target=y, ID="{}-{}".format(key[0], key[1]))
            else:
                return dict(input=X, ID="{}-{}".format(key[0], key[1]))
