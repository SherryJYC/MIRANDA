"""

Adaptation process.
source_model: pretrained PhenoFormer
target_model: not trained PhenoFormer

ADDA.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import torch.optim as optim
from model.discriminator import getDiscriminator
from adapters.data_utils import DomainAdaptationDataModule
from model.discriminator import getDiscriminator
from train import LitModel


class ADDAModel(LitModel):
    def __init__(self, 
                src_model,
                tgt_model,
                critic,
                target_scaler, target_list, 
                args, device, 
        ):
            super().__init__(
                backbone=src_model,
                target_scaler=target_scaler,
                args=args,
                device=device
                )
            
            self.save_hyperparameters(ignore=["src_model", "tgt_model", "critic", "device"])

            self.target_list = target_list

            self.src_domain_labels = torch.ones(args.batch_size).long().to(device)
            self.tgt_domain_labels = torch.zeros(args.batch_size).long().to(device)

            self.src_model = src_model
            self.tgt_model = tgt_model
            self.critic = critic

            self.n_task = src_model.backbone.n_task

            self.ce_loss = nn.CrossEntropyLoss()
            self.mse_loss = nn.MSELoss()

            # important for GAN-style training
            self.automatic_optimization = False

    
    def forward(self, batch, residual=False):
        # use encoder of tgt_model
        feature, _ = self.tgt_model.backbone.forward_features(batch)

        # use decoder of src_model
        task_embeddings = feature[:, : self.n_task, :]  # batch x n_task x d_model

        # use decoder of src_model
        preds = self.src_model.backbone.linear_decoder(task_embeddings)

        predictions = {
            self.target_list[i]: chunk[:, i, :].squeeze(1)
            for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
        }

        output = {"predictions": predictions}
        return output


    def training_step(self, batch, batch_idx):
        optimizer_tgt, optimizer_critic = self.optimizers()

        batch_src, batch_tgt = batch["source_domain"], batch["target_domain"]

        ############################
        # Train Critic
        ############################
        optimizer_critic.zero_grad()

        feat_src, _ = self.src_model.backbone.forward_features(batch_src)
        feat_src = feat_src[:, : self.n_task, :]

        feat_tgt, _ = self.tgt_model.backbone.forward_features(batch_tgt)
        feat_tgt = feat_tgt[:, : self.n_task, :]

        if self.args.gan_loss_type == "gan":
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            label_concat = torch.cat((self.src_domain_labels, self.tgt_domain_labels), 0)

            pred_concat = self.critic(feat_concat.detach())
            loss_critic = self.ce_loss(pred_concat, label_concat)

            pred_cls = torch.argmax(pred_concat, dim=1)
            acc = (pred_cls == label_concat).float().mean()
        elif self.args.gan_loss_type == "yeargan":
            pred_src = self.critic(feat_src.detach())
            pred_tgt = self.critic(feat_tgt.detach())

            loss_critic = (
                self.mse_loss(pred_src, batch_src["year_normalised"]) +
                self.mse_loss(pred_tgt, batch_tgt["year_normalised"])
            )
            acc = torch.tensor(0.0, device=self.device)

        else:
            raise ValueError(f"Unknown loss type {self.args.gan_loss_type}")

        self.manual_backward(loss_critic)
        optimizer_critic.step()

        ############################
        # Train Target Encoder
        ############################
        optimizer_tgt.zero_grad()

        feat_tgt, _ = self.tgt_model.backbone.forward_features(batch_tgt)
        feat_tgt = feat_tgt[:, : self.n_task, :]

        pred_tgt = self.critic(feat_tgt)

        label_fake = torch.ones(feat_tgt.size(0), dtype=torch.long, device=self.device)

        if self.args.gan_loss_type == "gan":
            loss_tgt = self.ce_loss(pred_tgt, label_fake)
        elif self.args.gan_loss_type == "yeargan":
            loss_tgt = -self.mse_loss(pred_tgt, batch_tgt["year_normalised"])
        else:
            raise ValueError(f"Unknown loss type {self.args.gan_loss_type}")

        self.manual_backward(loss_tgt)
        optimizer_tgt.step()

        ############################
        # Logging
        ############################
        losses = {
            "train/d_loss": loss_critic,
            "train/g_loss": loss_tgt,
            "train/critic_acc": acc
        }
        self.log_dictionary(losses, on_step=True)

        return loss_tgt
    
    def on_validation_epoch_start(self):
        self.meter_reset()
    
    def configure_optimizers(self):
        optimizer_tgt = optim.Adam(
            self.tgt_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.wd
        )

        optimizer_critic = optim.Adam(
            self.critic.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.wd
        )
        return [optimizer_tgt, optimizer_critic]

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
    output_dims = 2
    input_dims = args.d_model*n_task
    print(f"> input_dims of discriminator = {input_dims}")

    critic = getDiscriminator(input_dims=input_dims,
                              d_model=args.d_model, n_task=n_task, 
                              output_dims=output_dims,
                              use_softmax=True if loss_type == "gan" or "cls" in loss_type else False,
                              discriminator_type=args.discriminator_type,
                              ).to(device)

    # data module
    dm = DomainAdaptationDataModule(
        source_dataset=datasets["train_aug"],
        target_dataset=ConcatDataset([datasets["val"], datasets["test"]]),  # for target during training
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        batch_size=args.batch_size,
        train_loader_residual = datasets["train_loader_residual"]
    )

    model = ADDAModel(
        src_model=model["src_model"],
        tgt_model=model["tgt_model"],
        critic=critic,
        target_scaler=target_scaler,
        target_list=model["src_model"].backbone.target_list,
        args=args, device=device, 
    )

    xp_name = "ADDA"
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
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu" if args.gpus else "auto",
        devices=args.gpus if args.gpus else "auto",
        callbacks=[early_stop, ckpt],
        log_every_n_steps=5,
        max_epochs=args.adapt_epochs,
    )
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