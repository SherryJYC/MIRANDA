"""

Adaptation based on paper 'Unsupervised Domain Adaptation by Backpropagation' (https://arxiv.org/pdf/1409.7495)

"""
import numpy as np
import math
from pathlib import Path
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from model.discriminator import getDiscriminator
from model.dann import DANN
from configs.PROBLEM_CONFIG import target_shorter
from adapters.data_utils import DomainAdaptationDataModule, PairedDataset
from adapters.loss import RnCLoss, DomainContrastiveMemory, coral
from train import LitModel

class DomainAdaptationModel(LitModel):
    def __init__(self, backbone, target_scaler, target_list, 
                 len_train_loader,
                 args, device, 
                 dynamic_alpha=True,
                 alpha_schedule="dann", 
    ):
        super().__init__(
            backbone=backbone,
            target_scaler=target_scaler,
            args=args,
            device=device
            )
        
        self.save_hyperparameters()

        self.target_list = target_list

        self.src_domain_labels = torch.ones(args.batch_size).long().to(device)
        self.tgt_domain_labels = torch.zeros(args.batch_size).long().to(device)

        self.alpha = 0.5
        self.alpha_scheduler = AlphaScheduler(schedule=alpha_schedule, 
                                              total_steps=args.adapt_epochs * len_train_loader)
        self.dynamic_alpha = dynamic_alpha
        alpha_msg = f"> using alpha scheduler {alpha_schedule}" if dynamic_alpha else f"> using fixed alpha = {self.alpha}"
        print(alpha_msg)

        self.domain = "target"
        self.loss = args.loss

        if self.args.gan_loss_type == "rank":
            self.rank_criterion = RnCLoss(temperature=self.args.rank_temperature, feature_sim=self.args.rank_feature_sim)
            print(f">> using rank label {self.args.rank_label}, with temperature = {self.args.rank_temperature} \n feature_sim = {self.args.rank_feature_sim}")
            if self.args.use_memory:
                print(f">> using rank memory bank, mem_bank_size={self.args.mem_bank_size}")
                self.memorybank = DomainContrastiveMemory(dim=self.args.d_model, size=self.args.mem_bank_size)
        if self.args.use_rank_pheno:
            self.rank_pheno_criterion = RnCLoss(temperature=self.args.rank_temperature, label_diff='cos', feature_sim='l1')

    def forward(self, batch):
        out, _ = self.backbone(batch=batch, alpha=self.alpha, domain=self.domain) # {"predictions": predictions, "variances": variances}, domain_out
        return out
    
    def forward_views(self, data, domain):
        doms = []
        for view in ["original", "aug"]:
            reg, dom = self.backbone(
                batch=data[view],
                alpha=self.alpha,
                domain=domain
            )
            doms.append(dom)

        return reg, torch.cat(doms, dim=0)

    def training_step(self, batch, batch_idx):
        if self.dynamic_alpha:
            self.alpha = self.alpha_scheduler(step=self.global_step)

        source = batch["source_domain"]
        target = batch["target_domain"]

        if self.args.use_rank_pheno or self.args.use_CORAL: # here feat_src & feat_tgt (learnt tokens) are final features fed to regressor, not necessarily the features for DANN
            reg_src, domain_src, feat_src = self.backbone(batch=source, alpha=self.alpha, domain="source", return_features=True) # {"predictions": predictions, "variances": variances}, domain_out
            reg_tgt, domain_tgt, feat_tgt = self.backbone(batch=target, alpha=self.alpha, domain="target", return_features=True)
        else:
            reg_src, domain_src = self.backbone(batch=source, alpha=self.alpha, domain="source") # {"predictions": predictions, "variances": variances}, domain_out
            _, domain_tgt = self.backbone(batch=target, alpha=self.alpha, domain="target")

        # for source domain, labels are available
        # compute regression loss based on source domain
        if "NLL" in self.args.loss:
            loss_src_label, logvars = self.compute_loss_var(reg_src, source["target"], prefix="train")
        else:
            loss_src_label = self.compute_loss(reg_src["predictions"], source["target"], prefix="train")
            logvars = {}

        # metrics are based on source domain
        metrics = self.compute_batch_metrics(reg_src["predictions"], source["target"], prefix="train")

        # compute domain loss for source & target domain
        if self.args.gan_loss_type == "rank":
            rank_label = self.args.rank_label
            if hasattr(self, "memorybank"):
                 # update bank
                self.memorybank.update(features=domain_src.detach(), labels=source[rank_label].unsqueeze(1), domain="source")
                self.memorybank.update(features=domain_tgt.detach(), labels=target[rank_label].unsqueeze(1), domain="target")

                # get negative samples from bank
                bank_src = self.memorybank.get_bank(domain="source") # used as negative samples for target
                bank_tgt = self.memorybank.get_bank(domain="target")

                # compute loss
                loss_rank_src = self.rank_criterion(torch.cat((domain_src, bank_tgt["features"]), dim=0),
                                                torch.cat((source[rank_label].unsqueeze(1), bank_tgt["labels"]), dim=0))
                loss_rank_tgt = self.rank_criterion(torch.cat((domain_tgt, bank_src["features"]), dim=0),
                                                torch.cat((target[rank_label].unsqueeze(1), bank_src["labels"]), dim=0))
                loss_rank = 0.5 * (loss_rank_src + loss_rank_tgt)
            else:
                if self.args.rank_mask:
                    loss_rank = 0
                    # [B, n_task, d_model]
                    for i, t in enumerate(self.target_list):
                        valid = source["target"][t] != self.nan_value_target
                        loss_rank += self.rank_criterion(torch.cat((domain_src[:, i, :][valid], domain_tgt[:, i, :][valid]), dim=0),
                                                    torch.cat((source[rank_label][valid], target[rank_label][valid]), dim=0).unsqueeze(1))
                    loss_rank /= len(self.target_list)
                else:
                    loss_rank = self.rank_criterion(torch.cat((domain_src, domain_tgt), dim=0),
                                                    torch.cat((source[rank_label], target[rank_label]), dim=0).unsqueeze(1))
            domain_losses = {"train/L_rank": loss_rank}
            loss = loss_src_label["train/loss"] + loss_rank * self.args.rank_multiply
        elif self.args.gan_loss_type == "no_domain_loss":
            loss = loss_src_label["train/loss"]
            domain_losses = {}
        else:
            loss_src_domain = self.compute_gan_loss(domain_src, self.src_domain_labels, source)
            loss_tgt_domain = self.compute_gan_loss(domain_tgt, self.tgt_domain_labels, target)

            domain_losses = {"train/L_src_domain": loss_src_domain, "train/L_tgt_domain": loss_tgt_domain}

            loss = loss_src_label["train/loss"] + loss_src_domain + loss_tgt_domain

        # coral loss
        if self.args.use_CORAL:
            coral_loss = 0
            for i in range(len(self.target_list)):
                coral_loss += coral(feat_src[:, i, :], feat_tgt[:, i, :])
            loss_src_label["train/CORAL_loss"] = coral_loss / len(self.target_list)
            loss += loss_src_label["train/CORAL_loss"] * self.alpha

        # rank-phenology loss
        if self.args.use_rank_pheno:
            loss_rank_src = 0
            loss_rank_tgt = 0
            for i, t in enumerate(self.target_list):
                loss_rank_src += self.rank_criterion(feat_src[:, i, :].squeeze(), source["target"][t].unsqueeze(1))
            domain_losses[f"train/L_rank_pheno_src"] = loss_rank_src / self.backbone.n_task
            loss = loss + domain_losses[f"train/L_rank_pheno_src"]

        self.log_dictionary({**domain_losses, **loss_src_label, **metrics, **logvars}, on_step=True)
        
        if torch.isnan(loss):
            print("nan")
        return loss
    
    def compute_gan_loss(self, predictions, targets, inputs):
        if self.args.gan_loss_type == "gan":
            gan_loss = F.nll_loss(predictions, targets)
        elif "+" in self.args.gan_loss_type:
            gan_loss, start_idx = 0, 0
            for l in self.args.gan_loss_type.split("+"):
                num_dims = 2 if l == "latlon" else 1 # year, elevation has dim=1 while latlon has dim=2
                gan_loss += F.mse_loss(predictions[:, start_idx:start_idx+num_dims].squeeze(), inputs[l+"_normalised"].squeeze())
                start_idx = start_idx+num_dims
            gan_loss /= len(self.args.gan_loss_type.split("+")) # to make loss in a stable numeric range & comparable to reg loss
        elif self.args.gan_loss_type in ["elevation", "year", "latlon", "year_temp"]:
            gan_loss = F.mse_loss(predictions.squeeze(), inputs[self.args.gan_loss_type+"_normalised"].squeeze())
        elif self.args.gan_loss_type == "cls_year_temp":
            gan_loss = F.nll_loss(predictions, inputs["cls_year_temp"].long())
        else:
            raise
        return gan_loss

    
class AlphaScheduler:
    def __init__(
        self,
        total_steps: int,
        schedule: str = "dann",
        num_cycles: int = 2,
    ):
        """
        Args:
            total_steps: total number of training steps
            schedule: one of ['dann', 'cosine_restart',
                              'cosine_bump']
            num_cycles: number of cosine cycles (for cosine_restart)
        """
        self.total_steps = total_steps
        self.schedule = schedule.lower()
        self.num_cycles = num_cycles

    def __call__(self, step):
        t = step
        T = self.total_steps

        if self.schedule == "dann":
            p = float(step) / self.total_steps
            return 2. / (1. + np.exp(-10 * p)) - 1

        elif self.schedule == "cosine_restart":
            cycle_length = T // self.num_cycles
            cycle_progress = (t % cycle_length) / cycle_length
            return 0.5 * (1 - math.cos(2 * math.pi * cycle_progress))

        elif self.schedule == "cosine_bump":
            # One full cosine bump: 0 → 1 → 0
            return 0.5 * (1 - math.cos(2 * math.pi * t / T))

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    

def adaptation(model, 
               datasets,
               args, 
               device='cuda:0',
               target_scaler=None):
    # ------------
    # setup
    # ------------
    src_model = model["src_model"]
    loss_type = args.gan_loss_type
    n_task = src_model.backbone.n_task

    print(f'--- use loss type: {loss_type}')
    print(f"len(train)={len(datasets['train'])}, len(val)={len(datasets['val'])}, len(test)={len(datasets['test'])}")

    # init discriminator
    output_dims = 2 if loss_type == "gan" else len(loss_type.split("+"))
    if "latlon" in loss_type:
        output_dims += 1
    if loss_type == "rank":
        output_dims = args.rank_d_model
    elif "cls" in loss_type:
        output_dims = args.num_cls
    print(f"> output_dims of discriminator = {output_dims}")
    input_dims = args.d_model*n_task
    if args.pheno_model in ["InceptionFormer", "GRUFormer"] and not args.use_cross_attn:
        input_dims = args.d_model
    if args.rank_mask or args.avg_pool or args.critic_cross_attn:
        input_dims = args.d_model
    print(f"> input_dims of discriminator = {input_dims}")
    critic = getDiscriminator(input_dims=input_dims,
                              d_model=args.d_model, n_task=n_task, 
                              output_dims=output_dims,
                              use_softmax=True if loss_type == "gan" or "cls" in loss_type else False,
                              discriminator_type=args.discriminator_type,
                              rank_mask=args.rank_mask,
                              avg_pool=args.avg_pool,
                              cross_attn=args.critic_cross_attn).to(device)
    # init DANN model
    adaptation_model = DANN(base_model=src_model.backbone, critic=critic, args=args)

    dm = DomainAdaptationDataModule(
        source_dataset=datasets["train_aug"],
        target_dataset=ConcatDataset([datasets["val"], datasets["test"]]),  # for target during training
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        batch_size=args.batch_size,
    )

    model = DomainAdaptationModel(backbone=adaptation_model, target_scaler=target_scaler, 
                                  target_list=src_model.backbone.target_list, 
                                  len_train_loader=len(datasets["train"])//args.batch_size,
                                  args=args, device=device)
    
    xp_name = "adapt_dann"
    wandb_logger = WandbLogger(name=xp_name, save_dir=args.save_dir, offline=not args.wandb_online, project='phenocast')
    monitor_metric = "val/rmse"
    monitor_mode = "min"
    early_stop = EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=30)
    ckpt = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor=monitor_metric, mode=monitor_mode,
    )

    # ------------
    # train
    # ------------
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=[early_stop, ckpt],
                         log_every_n_steps=5,
                         max_epochs=args.adapt_epochs,
                         gpus=args.gpus)
    trainer.fit(model, datamodule=dm)
    # save best
    # wandb.save(ckpt.best_model_path)

    # ------------
    # test
    # ------------

    trainer.test(model=model, datamodule=dm, ckpt_path="best")

    metrics = wandb_logger.experiment.summary
    metrics_dict = dict(metrics)
    print(metrics_dict)
