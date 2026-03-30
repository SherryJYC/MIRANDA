import json
import os
import copy
import argparse
from argparse import ArgumentParser
from pathlib import Path
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from configs.PROBLEM_CONFIG import target_list_parser, target_shorter, seeds
from dataset import ClimatePhenoDataset, get_matching_indices
from adapters.data_utils import CombinedLoader
from model import architecture

class LitModel(pl.LightningModule):
    """PyTorch Lighnting module used as wrapper for the PhenoFormer model
    This module implements the functions that are used for model training
    and metric computations.
    """

    def __init__(
        self,
        backbone,
        target_scaler,
        args,
        device,
        output_dir="output/csv"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.target_scaler = target_scaler
        self.target_list = backbone.target_list
        self.nan_value_target = args.nan_value_target
        self.args = args

        if args.loss == "L2":
            self.loss_func = F.mse_loss
        elif args.loss == "L1":
            self.loss_func = F.l1_loss
        elif args.loss == "GNLL":
            self.loss_func = F.gaussian_nll_loss
        else:
            raise
        self.loss = args.loss

        self.to(device)

        self.meters = {
            "rmse": {t: MeanSquaredError(squared=False) for t in self.target_list},
            "r2": {t: R2Score() for t in self.target_list},
            "mae": {t: MeanAbsoluteError() for t in self.target_list},
        }
        for m in self.meters.keys():
            for t in self.target_list:
                self.meters[m][t].to(self.device)

        if args.save_test_results:
            self.test_outputs = defaultdict(list)
            self.test_output_file = Path(output_dir) / f"{args.unique_id}.csv"

    def forward(self, batch, residual=False):
        if residual:
            y_hat = self.backbone.forward_residual(batch)
        else:
            y_hat = self.backbone(batch)
        return y_hat
    
    def predict_unnormalised_dates(self, batch):
        y_hat = self.forward(batch=batch)
        for t in self.target_list:
            normalised_prediction = y_hat[t]
            mean, std = self.target_scaler[t]
            unnormalised_prediction = normalised_prediction * std + mean
            y_hat[t] = unnormalised_prediction
        return y_hat

    def meter_forward(self, predictions, targets, select_dim=None):
        if "variances" in predictions:
            predictions = predictions["predictions"]
        values = {
            m: {} for m in self.meters.keys() if m != "r2"
        }  # computing r2 on batch doesn't make sense
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]
            valid = true != self.nan_value_target
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            preds = preds * scaler
            true = true * scaler
            for m in self.meters.keys():
                if m == "r2":
                    self.meters[m][t].update(preds[valid], true[valid])
                else:
                    values[m][t] = self.meters[m][t](preds[valid], true[valid])
        return values

    def meter_update(self, predictions, targets, select_dim=None):
        if "variances" in predictions:
            predictions = predictions["predictions"]
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]
            valid = true != self.nan_value_target
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            preds = preds * scaler
            true = true * scaler
            for m in self.meters.keys():
                self.meters[m][t].update(preds[valid], true[valid])

    def meter_compute(self):
        values = {m: {} for m in self.meters.keys()}
        for t in self.target_list:
            for m in self.meters.keys():
                try:
                    values[m][t] = self.meters[m][t].compute()
                except ValueError:
                    values[m][t] = torch.tensor(torch.nan, device=self.device)
        return values

    def meter_reset(self):
        for t in self.target_list:
            for m in self.meters.keys():
                self.meters[m][t].reset()

    def multi_apply(
        self, function, predictions, targets, rescale=False, select_dim=None
    ):
        values = {}
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]

            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            valid = true != self.nan_value_target
            if rescale:
                values[t] = function(preds[valid], true[valid], scaler=scaler)
            else:
                values[t] = function(preds[valid], true[valid])
        return values
    
    def multi_apply_var(
        self, function, predictions, targets, rescale=False, select_dim=None
    ):
        variances, predictions = predictions["variances"], predictions["predictions"]

        values = {}
        logvars = {}
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            vars = (
                variances[t] if select_dim is None else variances[t][:, select_dim]
            ).exp() # model predicts log_var for stability
            true = targets[t]
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            valid = true != self.nan_value_target
            if rescale:
                values[t] = function(preds[valid], true[valid], vars[valid], scaler=scaler)
            else:
                values[t] = function(preds[valid], true[valid], vars[valid])
            logvars[t] = vars[valid].mean()
        return values, logvars

    def compute_batch_metrics(self, predictions, targets, prefix=""):
        out = {}
        meter_output = self.meter_forward(predictions, targets)
        for metric_name, metric_vals in meter_output.items():
            metric_vals = {
                f"{prefix}/{metric_name}_{target_shorter(k)}": v
                for k, v in metric_vals.items()
            }
            metric_vals[f"{prefix}/{metric_name}"] = torch.stack(
                [v for v in metric_vals.values() if not torch.isnan(v)]
            ).mean()
            out = {**out, **metric_vals}
        return out

    def compute_epoch_metrics(self, prefix=""):
        out = {}
        meter_output = self.meter_compute()
        for metric_name, metric_vals in meter_output.items():
            metric_vals = {
                f"{prefix}/{metric_name}_{target_shorter(k)}": v
                for k, v in metric_vals.items()
            }
            metric_vals[f"{prefix}/{metric_name}"] = torch.stack(
                [v for v in metric_vals.values() if not torch.isnan(v)]
            ).mean()
            out = {**out, **metric_vals}
        return out

    def compute_loss(self, predictions, targets, prefix=""):
        loss_vals = self.multi_apply(self.loss_func, predictions, targets)
        loss_vals = {
            f"{prefix}/loss_{target_shorter(k)}": v for k, v in loss_vals.items()
        }
        loss_vals[f"{prefix}/loss"] = torch.stack(
            [v for v in loss_vals.values() if not torch.isnan(v)]
        ).mean(0)
        return loss_vals
    
    def compute_loss_residual(self, predictions, batch, prefix=""):
        label_src, label_tgt = batch["source_domain"]["target"], batch["target_domain"]["target"]
        label_residual = {}
        for t in self.target_list:
            label_residual[t] = label_tgt[t] - label_src[t]
            invalid = (label_tgt[t] == self.nan_value_target) | (label_src[t] == self.nan_value_target)
            label_residual[t][invalid] = self.nan_value_target # will be filtered out in loss computation
        loss_vals = self.compute_loss(predictions, label_residual, prefix)
        return loss_vals
    
    def compute_loss_var(self, predictions, targets, prefix=""):
        loss_vals, logvar_vals = self.multi_apply_var(self.loss_func, predictions, targets)
        loss_vals = {
            f"{prefix}/loss_{target_shorter(k)}": v for k, v in loss_vals.items()
        }
        logvar_vals = {
            f"{prefix}/logvar_{target_shorter(k)}": v for k, v in logvar_vals.items()
        }

        loss_vals[f"{prefix}/loss"] = torch.stack(
            [v for v in loss_vals.values() if not torch.isnan(v)]
        ).mean(0)
        return loss_vals, logvar_vals

    def log_dictionary(self, dct, **kwargs):
        for name, value in dct.items():
            if not torch.isnan(value):
                self.log(name, value, kwargs)

    def on_fit_start(
        self,
    ):
        if self.device.type == "cuda":
            for m, d in self.meters.items():
                for t in self.target_list:
                    d[t].cuda()

    def training_step(self, batch, batch_idx):
        if self.args.residual:
            y_hat = self.forward(batch=batch, residual=True)
            y = batch["target_domain"]["target"]
        else:
            y_hat = self.forward(batch=batch) # y_hat = {"predictions": predictions, "variances": variances}
            y = batch["target"]

        # uncertainty estimation
        if "NLL" in self.args.loss:
            losses, logvars = self.compute_loss_var(y_hat, y, prefix="train")
        else:
            losses = self.compute_loss(y_hat["predictions"], y, prefix="train")
            logvars = {}

        loss = losses["train/loss"]
        
        metrics = self.compute_batch_metrics(y_hat["predictions"], y, prefix="train")
        self.log_dictionary({**losses, **metrics, **logvars}, on_step=True)

        if torch.isnan(loss):
            print("nan")
        return loss

    def validation_step(self, batch, batch_idx):
        if "target_domain" in batch.keys():
            y_hat = self.forward(batch=batch["target_domain"])
            y = batch["target_domain"]["target"]
        else:
            y_hat = self.forward(batch=batch) # y_hat = {"predictions": predictions, "variances": variances}
            y = batch["target"]

        if "NLL" in self.args.loss:
            losses, logvars = self.compute_loss_var(y_hat, y, prefix="val")
        else:
            losses = self.compute_loss(y_hat["predictions"], y, prefix="val")
            logvars = {}

        self.meter_update(y_hat["predictions"], y)
        self.log_dictionary({**losses, **logvars}, on_epoch=True)

    def test_step(self, batch, batch_idx):
        if "target_domain" in batch.keys():
            y_hat = self.forward(batch=batch["target_domain"])
            y_hat_residual = self.forward(batch=batch, residual=True) # pred_src + pred_residual
            y = batch["target_domain"]["target"]
            year = batch["target_domain"]["year"]
            elevation = batch["target_domain"]["elevation"]
        else:
            y_hat = self.forward(batch=batch) # y_hat = {"predictions": predictions, "variances": variances}
            y = batch["target"]
            year = batch["year"]
            elevation = batch["elevation"]

        if "NLL" in self.args.loss:
            losses, logvars = self.compute_loss_var(y_hat, y, prefix="test")
        else:
            losses = self.compute_loss(y_hat["predictions"], y, prefix="test")
            logvars = {}

        self.meter_update(y_hat["predictions"], y)
        self.log_dictionary({**losses, **logvars}, on_epoch=True)

        # save to results
        if hasattr(self, "test_outputs"):
            for k in self.target_list:
                m, s = self.target_scaler[k]
                pred = y_hat["predictions"][k] * s + m
                label = y[k] * s + m
                self.test_outputs[f"pred_{k}"].extend(pred.cpu().tolist())
                self.test_outputs[f"true_{k}"].extend(label.cpu().tolist())
            self.test_outputs["year"].extend(year.cpu().tolist())
            self.test_outputs["elevation"].extend(elevation.cpu().tolist())
            if "target_domain" in batch.keys():
                for k in self.target_list:
                    m, s = self.target_scaler[k]
                    pred = y_hat_residual["predictions"][k] * s + m
                    true_source = batch["source_domain"]["target"][k] * s + m
                    self.test_outputs[f"pred_residual_{k}"].extend(pred.cpu().tolist())
                    self.test_outputs[f"true_source_{k}"].extend(true_source.cpu().tolist())

                    if "predictions_residual" in y_hat_residual.keys():
                        pred_residual = y_hat_residual["predictions_residual"][k] * s
                        self.test_outputs[f"pred_residual_residual_{k}"].extend(pred_residual.cpu().tolist())
                    if "predictions_src" in  y_hat_residual.keys():
                        pred_src = y_hat_residual["predictions_src"][k] * s + m
                        self.test_outputs[f"pred_src_{k}"].extend(pred_src.cpu().tolist())
                    

    def on_validation_epoch_start(self):
        if self.global_step > 0:
            metrics = self.compute_epoch_metrics(prefix="train")
            self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

    def on_validation_epoch_end(self):
        metrics = self.compute_epoch_metrics(prefix="val")
        self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

    def on_test_epoch_end(self):
        metrics = self.compute_epoch_metrics(prefix="test")
        self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

        if hasattr(self, "test_outputs"):
            df = pd.DataFrame(self.test_outputs)
            df.to_csv(self.test_output_file, index=False)
            print(f">>>> saved test inference results into {self.test_output_file}")

    def on_test_epoch_start(self):
        self.meter_reset()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if self.args.optim == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd
            )
        elif self.args.optim == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd
            )
        else:
            raise "Unkown optimizer"


def get_parser():
    # ------------
    # The present script is used to run all the different configurations
    # of PhenoFormer presented in the paper. The following menu enables
    # the selection of a given model and data configuration
    # ------------
    parser = ArgumentParser()

    # Dataset and data splitting args
    parser.add_argument("--data_folder", default=None, type=str, help="path to the dataset folder")
    parser.add_argument("--target", type=str, default=None, help="target to predict (see target_list_parser in configs/PROBLEM_CONFIG.py for details)")
    parser.add_argument("--nan_value_target", type=int, default=-1, help="value used to represent missing target values")
    parser.add_argument("--nan_value_climate", type=int, default=0, help="value used to represent missing climate values")
    parser.add_argument("--sigma_jitter", default=0, type=float, help="standard deviation of the gaussian noise added to the input climate data")
    parser.add_argument("--split_mode", default="structured", type=str, help="mode used to split the dataset (see datasplit_configs in configs/RUN_CONFIGS.py for details)")
    parser.add_argument("--train_years_to", default=2002, type=int, help="last year of the training set (used in `structured` split mode)")
    parser.add_argument("--val_years_to", default=2012, type=int, help="last year of the validation set (used in `structured` split mode)")
    parser.add_argument("--input_phases", default=None, type=str, help="list of phenophases used as input (used of Variant f of PhenoFormer)")

    # Model args
    parser.add_argument("--nhead", default=8, type=int, help="number of heads in the transformer layer of PhenoFormer")
    parser.add_argument("--d_model", default=64, type=int, help="dimension of the inner representations of PhenoFormer")
    parser.add_argument("--n_layers", default=1, type=int, help="number of stacked attention layers in PhenoFormer")
    parser.add_argument("--dim_feedforward", default=128, type=int, help="number of neurons in the feedforward layer of PhenoFormer")
    parser.add_argument("--T_pos_enc", default=1000, type=int, help="maximal period used in the positional encoding of PhenoFormer")
    parser.add_argument("--elevation", dest="elevation", action="store_true", help="(flag) if set the elevation of the observation site is used as additional input")
    parser.add_argument("--latlon", dest="latlon", action="store_true", help="(flag) if set the latitude and longitude of the observation site are used as additional input")
    parser.add_argument("--norm_type", default="layernorm", type=str, help="type of norm layer of transformer encoder in PhenoFormer")
    parser.add_argument("--pheno_model", default="PhenoFormer", type=str, help="name of model used, [PhenoFormer, PhenoFormerReconstruct]")
    parser.add_argument("--gated_attn", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shallow", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--residual", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--norm_first", default=False, action=argparse.BooleanOptionalAction)

    # Training args
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--optim", type=str, default="adam", help="optimizer")
    parser.add_argument("--loss", default="L2", type=str, help="loss function (L1 or L2)")
    parser.add_argument("--grad_clip", default=0, type=float, help="gradient clipping value")
    parser.add_argument("--cross_val_id", default=None, type=str,help="cross validation id, the same id will be used for all"
    " folds of a given configuration. This enables to easily compute cross-fold average performance.") 
    parser.add_argument("--fold", default=None, type=int, help="fold number")

    # Output and logging args
    parser.add_argument("--save_dir", default="./training_logs", type=str)
    parser.add_argument("--model_tag", type=str, default="v0", help="model tag (used for logging)")
    parser.add_argument("--config_tag", type=str, default=None, help="config tag (used for logging)")
    parser.add_argument("--task_tag", type=str, default=None, help="task tag (single/multi species) (used for logging)" )
    parser.add_argument("--run_tag", type=str, default=None, help="run tag (used for logging)")
    parser.add_argument("--xp_name", default=None, type=str, help="experiment name (used for logging)")
    parser.add_argument("--wandb_online", default=False, action=argparse.BooleanOptionalAction)

    # Test only setting
    parser.add_argument("--use_pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pretrained_weights_folder", default="pre-trained-weights", help="folder where .ckpt exists")

    # Adaptation setting
    parser.add_argument("--adapt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapt_from_scratch", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapter", default="dann", type=str, help="type of adapters [adda, dann, tta]")
    parser.add_argument("--adapt_epochs", default=100, type=int, help="num epochs of adaptation")
    parser.add_argument("--gan_loss_type", default="gan", type=str, help="type of loss function in GAN [gan, year]")
    parser.add_argument("--discriminator_type", default="mlp", type=str, help="type of discriminator [mlp, transformer]")
    parser.add_argument("--use_two_views", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_memory", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mem_bank_size", default=128, type=int, help="sample size of memory bank")
    parser.add_argument("--num_cls", default=15, type=int, help="num of classes if use classification loss in discriminator")
    parser.add_argument("--use_cross_attn", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rank_feature_sim", default="l2", type=str, help="type of feature similarity used in rank loss")
    parser.add_argument("--rank_label", default="year_normalised", type=str, help="label used for rank-based contrastive learning")
    parser.add_argument("--rank_temperature", type=float, default=0.1, help="temperature paremeter for rank loss")
    parser.add_argument("--use_rank_pheno", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_test_results", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapt_norm_type", default="layernorm", type=str, help="type of norm layer of transformer encoder (high-level one, not used for DA)")
    parser.add_argument("--adapt_norm_first", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--add_layernorm", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adaptive_norm", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rank_mask", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adaBN", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--avg_pool", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--critic_cross_attn", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_CORAL", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rank_multiply", type=float, default=2, help="rank loss multiplies this rate")
    parser.add_argument("--rank_d_model", type=int, default=128, help="d_model for emebedding used for rank loss")
    parser.add_argument("--load_M1", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--full_eval", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--unique_id", default=None, help="unique id of a run")


    parser.set_defaults(
         elevation=False, latlon=False,
    )
    parser = pl.Trainer.add_argparse_args(parser)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Set the random seed (the random seed used for each fold is fixed
    #  across configurations but different between folds)
    if args.fold is not None:
        pl.seed_everything(seeds[args.fold])
    else:
        pl.seed_everything(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------
    # Dataset 
    # ------------
    load_M1_info = None
    if args.load_M1:
        M1_split = ""
        if args.split_mode == "structured":
            split = "structured_temporal"
        elif "hotyear-temporal-split" in args.split_mode:
            split = "hotyear_temporal"
        elif "elevation-spatial-split" in args.split_mode:
            split = "highelevation_spatial"
        else:
            raise
        load_M1_info = {
            "pred_folder": "output/processed_models/preds",
            "split": split,
            "fold": args.fold if args.fold is not None else 1,
        }
        print(load_M1_info)
    dt_base_args = dict(
        folder=args.data_folder,
        target_list=target_list_parser(args.target),
        normalise_climate=True,
        normalise_dates=True,
        nan_value_climate=args.nan_value_climate,
        nan_value_target=args.nan_value_target,
        phases_as_input=target_list_parser(args.input_phases),
        load_M1_info=load_M1_info
    )
    # Dataset without augmentation
    dt = ClimatePhenoDataset(**dt_base_args)
    # Dataset with augmentation
    dt_augm = ClimatePhenoDataset(**dt_base_args, sigma_jitter=args.sigma_jitter)

    # ------------
    # Dataset splitting
    # ------------
    if args.split_mode.endswith(".json"):
        with open(args.split_mode) as file:
            split = json.loads(file.read())
        if "hotyear-temporal-split" in args.split_mode:
            print(">>> domain adaptation experiment: hotyear-temporal-split")
            train_idxs = get_matching_indices(dt.years, split["train"])
            val_idxs = get_matching_indices(dt.years, split["val"])
            test_idxs = get_matching_indices(dt.years, split["test"])
        elif "elevation-spatial-split" in args.split_mode:
            print(f">>> domain adaptation experiment: {args.split_mode}")
            train_idxs = get_matching_indices(dt.sites, split["train"])
            val_idxs = get_matching_indices(dt.sites, split["val"])
            test_idxs = get_matching_indices(dt.sites, split["test"])
        else:
            train_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["train"])
            val_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["val"])
            test_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["test"])
    elif args.split_mode == "structured":
        train_idxs = list(
            np.where(np.array(dt.years).astype(int) <= args.train_years_to)[0]
        )
        test_idxs = list(
            np.where(np.array(dt.years).astype(int) > args.val_years_to)[0]
        )
        val_idxs = list(set(range(len(dt.years))) - set(train_idxs) - set(test_idxs))
    else:
        raise "Unknown split mode"

    train_loader = DataLoader(
        Subset(dt_augm, train_idxs),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dt, val_idxs),
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dt, test_idxs),
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    if args.residual:
        print(">>> using residual learning")
        train_len = len(train_idxs)
        train_loader1 = DataLoader(
            Subset(dt_augm, train_idxs), #[train_len//3:]),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8,
            shuffle=True,
        )
        train_loader2 = DataLoader(
            Subset(dt_augm, train_idxs[::-1]), #[:train_len//3]),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8,
            shuffle=True,
        )
        val_loader = CombinedLoader(source_loader=train_loader, target_loader=val_loader, cycle=False)
        test_loader = CombinedLoader(source_loader=train_loader, target_loader=test_loader, cycle=False)
        train_loader = CombinedLoader(source_loader=train_loader1, target_loader=train_loader2)

    print(f"dataset length: len(train)={len(train_loader)}, len(val)={len(val_loader)}, len(test)={len(test_loader)}")

    # ------------
    # model
    # ------------
    model_class = getattr(architecture, args.pheno_model)
    backbone = model_class(
        target_list=dt.target_list,
        d_in=len(dt.var_names),
        d_out=1,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        n_layers=args.n_layers,
        elevation=args.elevation,
        latlon=args.latlon,
        T_pos_enc=args.T_pos_enc,
        phases_as_input=target_list_parser(args.input_phases),
        norm_type=args.norm_type,
        use_nll="NLL" in args.loss,
        gated_attn=args.gated_attn,
        use_cross_attn=args.use_cross_attn,
        residual=args.residual,
        norm_first=args.norm_first,
        load_M1=args.load_M1,
    )

    model = LitModel(backbone=backbone, target_scaler=dt.target_scaler, args=args,
                     device=device)

    # ------------
    # training
    # ------------

    tags = [
        f"target/{args.target}",
        f"model/{args.model_tag}",
        f"run/{args.run_tag}",
        f"task/{args.task_tag}",
        f"config/{args.config_tag}",
    ]
    if args.fold is not None:
        tags.append(f"fold_{args.fold}")

    if args.cross_val_id is not None:
        name = f"{args.cross_val_id}_F{args.fold}"
    else:
        if args.xp_name is None:
            raise "If not running a N-fold cv, please specify the --xp_name argument"
        name = args.xp_name

    wandb_logger = WandbLogger(name=name, save_dir=args.save_dir, offline=not args.wandb_online, project='phenocast')
    logger = CSVLogger(save_dir=args.save_dir, name=name)

    monitor_metric = "val/rmse"
    monitor_mode = "min"
    early_stop = EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=30)
    ckpt = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor=monitor_metric, mode=monitor_mode,
    )
    trainer = pl.Trainer(
        logger=None if args.adapt else wandb_logger,
        gpus=args.gpus,
        callbacks=[early_stop, ckpt],
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clip,
    )

    if not args.adapt_from_scratch: # do normal training before adaptation
        if args.use_pretrained:
            assert args.pretrained_weights_folder is not None, "pretrained_weights_folder cannot be None!"
            print(f"================ Load pretrained weights from {args.pretrained_weights_folder }")

            # load pretrained model
            weight_files = [f for f in os.listdir(args.pretrained_weights_folder) if f.endswith('.ckpt')]
            weight_path = Path(args.pretrained_weights_folder) / weight_files[0]
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.target_scaler = checkpoint['hyper_parameters']['target_scaler']
            model.eval()
        else:
            trainer.fit(model, train_loader, val_loader)
            # wandb.save(ckpt.best_model_path)

        # ------------
        # testing
        # ------------
        if args.use_pretrained:
            result = trainer.test(model=model, dataloaders=test_loader)
        else:
            result = trainer.test(
                model=model, dataloaders=test_loader, ckpt_path=ckpt.best_model_path
            )
            best_path = ckpt.best_model_path
            print(f">> best ckpt path {best_path}")
            model.load_from_checkpoint(best_path)

            metrics = wandb_logger.experiment.summary
            metrics_dict = dict(metrics)

            output = {**vars(args), **metrics_dict}

            with open(
                Path(args.save_dir) / f"run_summary_fold{args.fold}.json", "w"
            ) as json_file:
                json.dump(output, json_file, indent=4)

        if args.full_eval:
            train_loader_eval = DataLoader(
                Subset(dt, train_idxs),
                batch_size=args.batch_size,
                drop_last=False,
                num_workers=8,
                shuffle=False,
            )
            for (split, ds) in zip(['train', 'val'], [train_loader_eval, val_loader]):
                if args.use_pretrained:
                    result = trainer.test(model=model, dataloaders=ds)
                else:
                    result = trainer.test(
                        model=model, dataloaders=ds, ckpt_path=ckpt.best_model_path
                    )
                print(f'------------------ {split} ------------------')
                print(result)

    # ------------
    # adapt
    # ------------
    if args.adapt:
        print("+++++++++++++++++ Adapdation +++++++++++++++++")
        model.to(device)
        tgt_model = copy.deepcopy(model)

        adapter_module = importlib.import_module(f"adapters.{args.adapter}")
        adapter_module.adaptation(
            args=args,
            device=model.device,
            target_scaler=dt.target_scaler, 
            model={"src_model": model, "tgt_model": tgt_model},
            datasets={"train_aug": Subset(dt_augm, train_idxs),
                      "val_aug": Subset(dt_augm, val_idxs),
                      "test_aug": Subset(dt_augm, test_idxs),
                      "train": Subset(dt, train_idxs),
                      "val": Subset(dt, val_idxs),
                      "test": Subset(dt, test_idxs),
                      "train_loader_residual": train_loader if args.residual else None
                      }
        )