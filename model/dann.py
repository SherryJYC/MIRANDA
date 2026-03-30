"""
DANN model of paper 'Unsupervised Domain Adaptation by Backpropagation' (https://arxiv.org/pdf/1409.7495)

codes adapted from https://github.com/fungtion/DANN/blob/master/models/model.py
"""

import copy
import torch
import torch.nn as nn
from torch.autograd import Function
from model.transformer_pytorch import TransformerEncoderLayer
from model.architecture import InceptionFormer, GRUFormer
from model.discriminator import DiscriminatorTrans
from adapters.norms import DomainAgnosticLayerNorm, AdaptiveBatchNorm1d

class DANN(nn.Module):
    def __init__(self, 
                 base_model, 
                 critic, 
                 args,
                 regression_only=False, 
                 ):
        super(DANN, self).__init__()

        self.base_model = base_model
        self.n_task = base_model.n_task
        self.target_list = base_model.target_list

        self.critic = critic
        
        self.regression_only = regression_only

        if args.shallow and not isinstance(self.base_model, InceptionFormer):
            # self.another_transformer = copy.deepcopy(self.base_model.transformer)
            self.another_transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=args.d_model,
                    nhead=args.nhead,
                    dim_feedforward=args.dim_feedforward,
                    batch_first=True,
                    norm_type=args.adapt_norm_type,
                    gated_attn=args.gated_attn,
                    norm_first=args.adapt_norm_first,
                    adaptive_norm=args.adaptive_norm,
                )
                for _ in range(args.n_layers)
            ]
            )
            print("> using shallow DANN")

        if args.add_layernorm:
            print(">> add DA layer norm before linear decoder")
            self.layernorm = DomainAgnosticLayerNorm(normalized_shape=args.d_model)
        
        if args.adaBN:
            print(">> add adaBN layer norm before linear decoder")
            self.adaBatchNorm = AdaptiveBatchNorm1d(args.d_model)


    def forward(self, batch, return_attention=False, alpha=0, domain="source", return_features=False, 
                only_features=False):

        if isinstance(self.base_model, InceptionFormer) or isinstance(self.base_model, GRUFormer):
            out, attention = self.base_model.forward_features(batch=batch, return_attention=return_attention, domain=domain, return_features=True)
            # out = {"transformer_features": out, "features": inception_out.permute(0, 2, 1)}
            features, domain_features = out["transformer_features"], out["features"] # [b, t, c], if use_cross_attn, domain features't = n_task
 
            if not isinstance(self.critic, DiscriminatorTrans) and not domain_features.shape[1] == self.n_task:
                # global avg pooling for MLP-based critic
                domain_features = domain_features.mean(dim=1, keepdim=True) # [b, 1, c]
        else:
            features, attention = self.base_model.forward_features(batch, return_attention, domain)
            domain_features = features[:, :self.n_task, :]

        # regression

        if hasattr(self, "another_transformer"):
            if return_attention:
                attentions = []
                for layer in self.another_transformer:
                    features, attention = layer(features, return_attention=True)
                    attentions.append(attention)
                attention = torch.stack(
                    attentions, 0
                )  # n_layer x B x n_head x target_sequence x source_sequence
            else:
                for layer in self.another_transformer:
                    features = layer(features, domain=domain)

        if only_features:
            return features
        
        if hasattr(self, "layernorm"):
            features = self.layernorm(features, domain=domain)
        if hasattr(self, "adaBatchNorm"):
            features = features[:, :self.n_task, :]
            features = self.adaBatchNorm(features.permute(0, 2, 1), domain=domain).permute(0, 2, 1)

        if domain == "target":
            # for rank_pheno loss, only regressor is optimized
            reg_predictions = self.forward_regression(features.detach(), batch)
        else:
            reg_predictions = self.forward_regression(features, batch)

        # discriminator
        reverse_features = ReverseLayerF.apply(domain_features, alpha)
        domain_predictions = self.critic(reverse_features)

        if self.regression_only:
            return reg_predictions

        if return_attention:
            return reg_predictions, domain_predictions, attention
        if return_features:
            return reg_predictions, domain_predictions, features[:, :self.n_task, :]
        return reg_predictions, domain_predictions
        
    def forward_regression(self, features, batch):
        if hasattr(self.base_model, "linear_encoder_M1"):
            m1preds = batch["M1_preds"] # dict
            m1tensor = batch["M1_tensors"].unsqueeze(-1) # [B, n_task, 1]
            m1tokens = self.base_model.linear_encoder_M1(m1tensor)
            features = torch.cat([features[:, : self.n_task, :], m1tokens], dim=1) # [B, n_task+n_task, d]
            preds = self.base_model.linear_decoder(features)[:, : self.n_task, :]

            predictions = {
                self.target_list[i]: chunk[:, i, :].squeeze(1) #+ m1preds[self.target_list[i]]
                    for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
            }
        else:
            # Linear decoder for each phenophase d_model -> 1 (or d_out)
            preds = self.base_model.linear_decoder(features[:, :self.n_task, :])

            predictions = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
            }
        out = {"predictions": predictions}

        if hasattr(self.base_model, "linear_decoder_var"):
            vars = self.base_model.linear_decoder_var(features[:, :self.n_task, :])
            variances = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(vars.chunk(self.n_task, dim=2))
            }
            out["variances"] = variances

        if hasattr(self.base_model, "decoder"):
            x_re = self.base_model.decoder(features[:, self.n_task:, :])
            out["reconstructed"] = x_re

        return out

    def forward_residual(self, batch, domain="source"):
        source, target = batch["source_domain"], batch["target_domain"]

        out_src = self.forward(source, only_features=True, domain=domain)
        task_embeddings_src = out_src[:, : self.n_task, :]
        out_tgt = self.forward(target, only_features=True, domain=domain)
        task_embeddings_tgt = out_tgt[:, : self.n_task, :]

        # no residual
        # preds = self.base_model.linear_decoder(task_embeddings_tgt)
        preds_src = self.base_model.linear_decoder(task_embeddings_src)

        # residual
        # preds_residual = preds - preds_src
        preds_residual = self.base_model.residual_linear_decoder(self.base_model.residual_cross_attn(query=task_embeddings_tgt, 
                                                                               key=task_embeddings_src, 
                                                                               value=task_embeddings_src)[0])

        predictions_src = {
            self.target_list[i]: chunk[:, i, :].squeeze(1)
            for i, chunk in enumerate(preds_src.chunk(self.n_task, dim=2))
        }

        # combine for final predictions
        predictions = {}
        for i, (preds_src_chunk, preds_residual_chunk) in enumerate(zip(preds_src.chunk(self.n_task, dim=2), 
                                                                    preds_residual.chunk(self.n_task, dim=2))):
            t = self.target_list[i]
            predictions[t] = preds_residual_chunk[:, i, :].squeeze(1) + preds_src_chunk[:, i, :].squeeze(1)

        output = {"predictions": predictions, "predictions_src": predictions_src}#, "predictions_residual": predictions_residual}
            
        return output


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None