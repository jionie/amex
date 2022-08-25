import os
import time
import sys
from joblib import dump, load
import numpy as np
import random
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss


from dataset.dataset import get_train_val_loader
from model.model import Model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]

    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]

    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


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
        self.min_lr = 4e-4
        self.warmup_lr = 2e-5
        self.weight_decay = 2e-2
        self.num_epoch = 100
        self.warmup_epoches = 1
        self.early_stopping = 20
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


class Trainer:
    def __init__(self, config):
        super(Trainer).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode="a+")

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = "cuda"
        else:
            self.config.device = "cpu"

    def load_data(self):

        self.start = time.time()
        self.log.write("\nLoading data...\n")

        self.train_data_loader, self.val_data_loader = get_train_val_loader(
            seed=self.config.seed,
            n_folds=5,
            target_fold=self.config.fold,
            batch_size=self.config.batch_size
        )

        self.train_data_loader_len = len(self.train_data_loader)

        self.eval_step = int(self.train_data_loader_len)
        self.log_step = int(self.train_data_loader_len)

        self.log.write("\nLoading data finished cost time {}s\n".format(time.time() - self.start))
        self.start = time.time()

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float("inf"), float("inf"), float("-inf")

        # eval setting
        self.eval_count = 0
        self.count = 0

    def pick_model(self):

        self.model = Model(
            cat_features_size=self.config.cat_features_size,
            num_features_size=self.config.num_features_size
        ).to(self.config.device)

    def differential_lr(self, warmup=True):

        param_optimizer = list(self.model.named_parameters())
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight"
        ]

        self.optimizer_grouped_parameters = []

        if warmup:
            self.optimizer_grouped_parameters.append(
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                            ],
                 "lr": self.config.warmup_lr,
                 "weight_decay": self.config.weight_decay,
                 }
            )
            self.optimizer_grouped_parameters.append(
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                            ],
                 "lr": self.config.warmup_lr,
                 "weight_decay": 0,
                 }
            )

        else:
            self.optimizer_grouped_parameters.append(
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                            ],
                 "lr": self.config.lr,
                 "weight_decay": self.config.weight_decay,
                 }
            )
            self.optimizer_grouped_parameters.append(
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                            ],
                 "lr": self.config.lr,
                 "weight_decay": 0,
                 }
            )

    def prepare_optimizer(self):

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.optimizer_grouped_parameters,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            self.config.min_lr,
            self.config.lr,
            step_size_up=int(self.train_data_loader_len),
            step_size_down=None,
            mode="triangular2",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=- 1,
            verbose=False
        )
        self.lr_scheduler_each_iter = True

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)

    def prepare_apex(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        checkpoint_to_save = {
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
        }

        save_path = os.path.join(self.config.checkpoint_folder, "{}.pth".format(np.around(self.valid_metric_optimal, 5)))
        torch.save(checkpoint_to_save, save_path)
        self.log.write("Model saved as {}.\n".format(save_path))

    def setup_model(self):

        self.log.write("\nSetting up model...\n")

        self.pick_model()

        self.differential_lr(warmup=True)

        self.prepare_optimizer()

        if self.config.apex:
            self.prepare_apex()

        self.log.write("\nSetting up model finished cost time {}s\n".format(time.time() - self.start))
        self.start = time.time()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def count_nonzero_parameters(self):
        # get total size of trainable parameters
        return sum(p.data.count_nonzero() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write("\n*General Setting*")
        self.log.write("\nseed: {}".format(self.config.seed))
        self.log.write("\ntrainable parameters:{:,.0f}".format(self.count_parameters()))
        self.log.write("\ndevice: {}".format(self.config.device))
        self.log.write("\nuse gpu: {}".format(self.config.use_gpu))
        self.log.write("\ndevice num: {}".format(self.num_device))
        self.log.write("\noptimizer: {}".format(self.optimizer))
        self.log.write("\nlr: {}".format(self.config.lr))
        self.log.write("\n")

    def train_batch(
            self,
            cat_features,
            num_features,
            target,
    ):
        # set input to cuda mode
        cat_features = cat_features.to(self.config.device, non_blocking=True).long()
        num_features = num_features.to(self.config.device, non_blocking=True).float()
        target = target.to(self.config.device, non_blocking=True).float()

        with torch.autograd.set_detect_anomaly(True):

            outputs = self.model(
                cat_features,
                num_features,
            ).squeeze()

        if self.config.apex:
            with torch.autograd.set_detect_anomaly(True):
                with torch.cuda.amp.autocast():
                    loss = self.criterion(outputs, target)

                self.scaler.scale(loss).backward()

        else:

            loss = self.criterion(outputs, target)
            loss.backward()

        return loss, outputs, target

    def train_op(self):

        self.show_info()
        self.log.write("** start training here! **\n")
        self.log.write("   batch_size=%d,  accumulation_steps=%d\n" % (self.config.batch_size,
                                                                       self.config.accumulation_steps))

        self.criterion = BCEWithLogitsLoss(reduction="mean")

        while self.epoch <= self.config.num_epoch:

            self.train_outputs = []
            self.train_targets = []

            # warmup lr for parameter warmup_epoch
            if self.epoch == self.config.warmup_epoches:
                self.differential_lr(warmup=False)
                self.prepare_optimizer()

            # update lr and start from start_epoch
            if (self.epoch >= 1) and (not self.lr_scheduler_each_iter):
                self.scheduler.step()

            self.log.write("\n")
            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write("\n")

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # set model training mode
            self.model.train()

            # init optimizer
            self.model.zero_grad()

            for tr_batch_i, (
                    cat_features,
                    num_features,
                    target,
            ) in enumerate(self.train_data_loader):

                batch_size = cat_features.shape[0]

                if tr_batch_i >= self.train_data_loader_len:
                    break

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group["lr"] / len(self.optimizer.param_groups)

                loss, outputs, target = self.train_batch(
                    cat_features,
                    num_features,
                    target
                )

                if (tr_batch_i + 1) % self.config.accumulation_steps == 0:
                    # use apex
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.optimizer.step()

                    self.model.zero_grad()

                    # adjust lr
                    if self.lr_scheduler_each_iter:
                        self.scheduler.step()

                    self.step += 1

                # translate to predictions
                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target = to_numpy(target)

                self.train_outputs.append(outputs)
                self.train_targets.append(target)

                sum_train_loss = sum_train_loss + np.array([loss.item() * batch_size])
                sum_train = sum_train + np.array([batch_size])

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0

                    train_outputs = np.concatenate(self.train_outputs, axis=0)
                    train_targets = np.concatenate(self.train_targets, axis=0)

                    train_metric = amex_metric(train_targets, train_outputs)

                    self.log.write(
                        "lr: {} loss: {} amex metric: {}\n"
                            .format(np.around(rate, 6),
                                    np.around(train_loss[0], 4),
                                    np.around(train_metric, 5)
                                    )
                    )

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.log.write("Training for one eval step finished cost time {}s\n\n".format(time.time() -
                                                                                                  self.start))
                    self.start = time.time()

                    self.evaluate_op()

                    self.log.write("Evaluating for one eval step finished cost time {}s\n\n".format(time.time() -
                                                                                                    self.start))
                    self.start = time.time()

                    self.model.train()

            if self.count >= self.config.early_stopping:
                break

            self.epoch += 1

    def evaluate_op(self):

        self.eval_count += 1
        self.criterion = BCEWithLogitsLoss(reduction="mean")

        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        self.eval_outputs = []
        self.eval_targets = []

        with torch.no_grad():
            for val_batch_i, (
                    cat_features,
                    num_features,
                    target,
            ) in enumerate(self.val_data_loader):

                batch_size = cat_features.shape[0]

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                cat_features = cat_features.to(self.config.device, non_blocking=True).long()
                num_features = num_features.to(self.config.device, non_blocking=True).float()

                target = target.to(self.config.device, non_blocking=True).float()

                outputs = self.model(
                    cat_features,
                    num_features,
                ).squeeze()

                loss = self.criterion(outputs, target)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target = to_numpy(target)

                self.eval_outputs.append(outputs)
                self.eval_targets.append(target)

                valid_loss = valid_loss + np.array([loss.item() * batch_size])
                valid_num = valid_num + np.array([batch_size])

            valid_loss = valid_loss / valid_num

            eval_outputs = np.concatenate(self.eval_outputs, axis=0)
            eval_targets = np.concatenate(self.eval_targets, axis=0)

            eval_metric = amex_metric(eval_targets, eval_outputs)

            self.log.write(
                "eval     loss:  {} amex metric: {}\n"
                    .format(
                    np.around(valid_loss[0], 4),
                    np.around(eval_metric, 5),
                )
            )

        if self.valid_metric_optimal <= eval_metric:
            self.valid_metric_optimal = eval_metric
            self.save_check_point()
            self.count = 0

        else:
            self.count += 1


if __name__ == "__main__":

    target_fold = 4
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
        fold=target_fold,
        cat_features_size=cat_features_size,
        num_features_size=len(num_features)
    )

    trainer = Trainer(config=config)

    trainer.train_op()
