import numpy as np
import pandas as pd
from joblib import dump, load
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


features = load("selected_features.pkl")

target = "target"

cat_features_base = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68"
]
cat_features = [
    "{}_last".format(feature) for feature in cat_features_base
]
cat_features = [feature for feature in cat_features if feature in features]

num_features = [feature for feature in features if feature not in cat_features]


class Dataset:
    def __init__(
        self,
        sample_indices,
        cat_features,
        num_features,
        target,
    ):

        self.sample_indices = sample_indices

        self.cat_features = cat_features
        self.num_features = num_features

        self.target = target

    def __getitem__(self, idx):

        # get feature
        cat_features = np.nan_to_num(self.cat_features[idx], nan=0, posinf=0, neginf=0)
        num_features = np.nan_to_num(self.num_features[idx], nan=0, posinf=0, neginf=0)

        # get target
        if self.target is not None:
            target = self.target[idx]
        else:
            target = None

        return cat_features, num_features, target

    def __len__(self):
        return len(self.sample_indices)


def collate(batch):

    cat_features = []
    num_features = []
    targets = []

    for (cat_feature, num_feature, target) in batch:
        cat_features.append(cat_feature)
        num_features.append(num_feature)
        targets.append(target)

    cat_features = torch.from_numpy(np.stack(cat_features)).contiguous().long()
    num_features = torch.from_numpy(np.stack(num_features)).contiguous().float()

    if targets[0] is None:
        targets = None
    else:
        targets = torch.from_numpy(np.stack(targets)).contiguous().float()

    return cat_features, num_features, targets


def get_train_val_loader(
    seed,
    n_folds,
    target_fold,
    batch_size=1024
):
    kfold = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    train = pd.read_parquet("../input/train_for_nn.parquet")

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[target])):

        if fold == target_fold:
            break

    train_cat_features = train.loc[trn_ind, cat_features].values
    train_num_features = train.loc[trn_ind, num_features].values
    train_target = train.loc[trn_ind, target].values

    train_dataset = Dataset(
        sample_indices=range(train_target.shape[0]),
        cat_features=train_cat_features,
        num_features=train_num_features,
        target=train_target,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate,
    )

    val_cat_features = train.loc[val_ind, cat_features].values
    val_num_features = train.loc[val_ind, num_features].values
    val_target = train.loc[val_ind, target].values

    val_dataset = Dataset(
        sample_indices=range(val_target.shape[0]),
        cat_features=val_cat_features,
        num_features=val_num_features,
        target=val_target,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate,
    )

    return train_loader, val_loader


def get_test_loader():

    test = pd.read_parquet("../input/test_for_nn.parquet")

    test_cat_features = test[cat_features].values
    test_num_features = test[num_features].values

    test_dataset = Dataset(
        sample_indices=range(test_cat_features.shape[0]),
        cat_features=test_cat_features,
        num_features=test_num_features,
        target=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate,
    )

    return test_loader
