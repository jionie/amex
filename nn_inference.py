import numpy as np
import random
import os
from joblib import load, dump
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss

from dataset.dataset import get_test_loader
from model.model import Model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    def __init__(
            self,
            seed,
            fold,
            cat_features_size,
            num_features_size,
    ):

        self.batch_size = 4096
        self.accumulation_steps = 1
        self.lr = 4e-3
        self.min_lr = 2e-4
        self.warmup_lr = 2e-5
        self.weight_decay = 2e-2
        self.num_epoch = 30
        self.warmup_epoches = 1
        self.early_stopping = 10
        self.max_grad_norm = 2
        self.seed = seed
        self.fold = fold
        self.cat_features_size = cat_features_size
        self.num_features_size = num_features_size

        self.checkpoint_path = "../ckpt/nn_seed_{}/".format(seed)
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.checkpoint_folder = os.path.join(self.checkpoint_path, "fold_{}".format(fold))
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)

        self.apex = True


if __name__ == "__main__":

    seed = 5120
    seed_everything(seed)

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

    cat_features_size = [4, 8, 3, 8, 3, 2, 6, 5, 3, 8]

    config = Config(
        seed=seed,
        fold=0,
        cat_features_size=cat_features_size,
        num_features_size=len(num_features)
    )

    test_data_loader = get_test_loader()
    preds = []

    for fold in range(5):

        model = Model(
            cat_features_size=config.cat_features_size,
            num_features_size=config.num_features_size
        ).cuda()
        model.eval()

        ckpt = torch.load(os.path.join(config.checkpoint_path, "fold_{}.pth".format(fold)))
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict)
        print("load from:", os.path.join(config.checkpoint_path, "fold_{}.pth".format(fold)))

        pred = None

        with torch.no_grad():
            for _, (
                cat_features,
                num_features,
                _,
            ) in enumerate(test_data_loader):

                # set input to cuda mode
                cat_features = cat_features.cuda().long()
                num_features = num_features.cuda().float()

                outputs = model(
                    cat_features,
                    num_features,
                ).squeeze()

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)

                if pred is None:
                    pred = outputs
                else:
                    pred = np.concatenate([pred, outputs], axis=0)

        preds.append(pred)

    preds = np.mean(preds, axis=0)
    dump(preds, "nn_pred.pkl")
