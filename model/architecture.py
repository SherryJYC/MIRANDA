import torch
import torch.nn as nn

from model.transformer_pytorch import TransformerEncoderLayer, TransformerDecoderLayer
from model.inception import InceptionBlock

class PhenoFormer(nn.Module):
    def __init__(
        self,
        target_list,
        d_in=7,
        d_out=1,
        d_model=64,
        nhead=8,
        dim_feedforward=128,
        n_layers=1,
        positional_encoding=True,
        elevation=False,
        latlon=False,
        T_pos_enc=1000,
        phases_as_input=None,
        norm_type="layernorm",
        norm_first=False,
        use_nll=False,
        gated_attn=False,
        use_cross_attn=False,
        residual=False,
        load_M1=False,
        **kwargs
    ):
        """Attention-based architecture for phenology modelling from
        climate time series.

        Args:
            target_list (list[str]): list of the names of the phenophases
            to be predicted, format: ["{species_name}:{phenophase_name}"]
            d_in (int, optional): Number of channels of the input time series 
            of climate variables. Defaults to 7.
            d_out (int, optional): Output dimension. Defaults to 1.
            d_model (int, optional): Dimension of the inner representations of the model.
            Defaults to 64.
            nhead (int, optional): Number of heads in the attention layer. Defaults to 8.
            dim_feedforward (int, optional): Number of neurons in the feedforward layer.
            Defaults to 128.
            n_layers (int, optional): Number of stacked attention layers. Defaults to 1.
            positional_encoding (bool, optional): If true, add positional encoding 
            to the input time series. Defaults to True.
            elevation (bool, optional): If true the elevation of the observaton site 
            is concatenated to the input data. Defaults to False.
            latlon (bool, optional): If true the geo-location of the observaton site 
            is concatenated to the input data. Defaults to False.
            T_pos_enc (int, optional): Maximal period used in the positional encoding. 
            Defaults to 1000.
            phases_as_input (list[str]): List of phenophase dates that 
            are given as input. Defaults to None.
        """
        super(PhenoFormer, self).__init__()

        self.positional_encoding = None
        if positional_encoding:
            self.positional_encoding = PositionalEncoder(d=d_model, T=T_pos_enc)
        self.target_list = target_list
        self.n_task = len(target_list) # number of phenophases to predict 
        print(f"> n_task = {self.n_task}")
        self.d_out = d_out
        self.elevation = elevation
        self.latlon = latlon
        self.add_static = elevation + latlon + (phases_as_input is not None)
        self.phases_as_input = phases_as_input

        # Dimension of the static features that are potentially concatenated to the input
        self.d_static = 1 * elevation + 2 * latlon
        if phases_as_input is not None:
            self.d_static += len(phases_as_input)

        # Shared Linear Encoder
        self.shared_linear_encoder = nn.Sequential(nn.Linear(d_in + self.d_static, d_model))

        # Learnt tokens for each phenophase to predict
        self.learnt_tokens = nn.ParameterDict(
            {
                t: nn.Parameter(torch.rand((1, d_model))).requires_grad_()
                for t in target_list
            }
        )

        # Transformer layer for temporal encoding 
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_type=norm_type,
                    gated_attn=gated_attn,
                    norm_first=norm_first
                )
                for _ in range(n_layers)
            ]
        )

        # Linear decoders (one per phenophase)
        self.linear_decoder = nn.Linear(in_features=d_model, out_features=d_out * self.n_task, bias=True)

        if load_M1:
            print(">> Hybrid model with M1 predictions")
            self.linear_encoder_M1 = nn.Linear(in_features=1, out_features=d_model, bias=True)

        if residual:
            self.residual_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
            self.residual_linear_decoder = nn.Linear(in_features=d_model, out_features=d_out * self.n_task, bias=True)

    def forward(self, batch, return_attention=False, domain="source"):
        out, attention = self.forward_features(batch, return_attention, domain)
        b = batch["climate"].size(0)
        
        # Decoding
        # Retrieve the embedding of each learnt token (each phenophase)
        task_embeddings = out[:, : self.n_task, :]  # batch x n_task x d_model

        if hasattr(self, "linear_encoder_M1"):
            m1preds = batch["M1_preds"] # dict
            m1tensor = batch["M1_tensors"].unsqueeze(-1) # [B, n_task, 1]
            m1tokens = self.linear_encoder_M1(m1tensor)
            task_embeddings = torch.cat([task_embeddings, m1tokens], dim=1) # [B, n_task+n_task, d]
            preds = self.linear_decoder(task_embeddings)[:, : self.n_task, :]

            predictions = {
                self.target_list[i]: chunk[:, i, :].squeeze(1) #+ m1preds[self.target_list[i]]
                    for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
            }
        else:
            # Linear decoder for each phenophase d_model -> 1 (or d_out)
            preds = self.linear_decoder(task_embeddings)

            predictions = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
            }

        output = {"predictions": predictions}

        if hasattr(self, 'linear_decoder_var'):
            vars = self.linear_decoder_var(task_embeddings)
            variances = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(vars.chunk(self.n_task, dim=2))
            }
            output["variances"] = variances

        if return_attention:
            output["attention"] = attention
            
        return output
    
    def forward_residual(self, batch, return_attention=False, domain="source"):
        source, target = batch["source_domain"], batch["target_domain"]

        out_src, attention = self.forward_features(source, return_attention, domain)
        task_embeddings_src = out_src[:, : self.n_task, :]
        out_tgt, attention = self.forward_features(target, return_attention, domain)
        task_embeddings_tgt = out_tgt[:, : self.n_task, :]

        # # no residual
        # preds = self.linear_decoder(task_embeddings_tgt)

        # # # residual
        # # preds_residual = preds - preds_src
        # preds_residual = self.residual_linear_decoder(self.residual_cross_attn(query=task_embeddings_tgt, 
        #                                                                        key=task_embeddings_src, 
        #                                                                        value=task_embeddings_src)[0])

        # # combine for final predictions
        # predictions = {}

        # predictions = {
        #     self.target_list[i]: chunk[:, i, :].squeeze(1)
        #     for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
        # }
        # predictions_residual = {
        #     self.target_list[i]: chunk[:, i, :].squeeze(1)
        #     for i, chunk in enumerate(preds_residual.chunk(self.n_task, dim=2))
        # }

        # output = {"predictions": predictions, "predictions_residual": predictions_residual}

        # no residual
        # preds = self.base_model.linear_decoder(task_embeddings_tgt)
        preds_src = self.linear_decoder(task_embeddings_src)

        # residual
        # preds_residual = preds - preds_src
        preds_residual = self.residual_linear_decoder(self.residual_cross_attn(query=task_embeddings_tgt, 
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

        if hasattr(self, 'linear_decoder_var'):
            vars = self.linear_decoder_var(task_embeddings_tgt)
            variances = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(vars.chunk(self.n_task, dim=2))
            }
            output["variances"] = variances

        if return_attention:
            output["attention"] = attention
            
        return output

    def forward_features(self, batch, return_attention=False, domain="source"):
        x = batch["climate"]
        b, t, c = x.shape

        # Concatenation of static features (if any)
        if self.d_static > 0:
            static_data = []
            if self.elevation:
                static_data.append(batch["elevation_normalised"].unsqueeze(1))
            if self.latlon:
                static_data.append(batch["latlon_normalised"])
            if self.phases_as_input is not None:
                for p in self.phases_as_input:
                    static_data.append(batch["input_phases"][p].unsqueeze(1))
            static_data = torch.cat(static_data, 1)
            x = torch.cat([x, static_data.unsqueeze(1).repeat(1, t, 1)], dim=2)
            b, t, c = x.shape

        # Shared linear encoder applied in parallel to all time steps 
        out = self.shared_linear_encoder(x.view(b * t, c)).view(b, t, -1)

        # Add positional encoding to encode the time information
        if self.positional_encoding is not None:
            positions = batch["doys"]
            if out.device != positions.device:
                print(f'----------- {out.device}, {positions.device}')
            out = out + self.positional_encoding(positions)

        # Prepend learnt tokens
        learnt_tokens = torch.cat([self.learnt_tokens[t] for t in self.target_list], 0)
        learnt_tokens = learnt_tokens.unsqueeze(0).repeat((b, 1, 1))
        out = torch.cat([learnt_tokens, out], dim=1)

        # Apply transformer layer
        if return_attention:
            if isinstance(self.encoder, nn.ModuleList):
                attentions = []
                for layer in self.transformer:
                    out, attention = layer(out, return_attention=True)
                    attentions.append(attention)
                attention = torch.stack(
                    attentions, 0
                )  # n_layer x B x n_head x target_sequence x source_sequence
            else:
                out, attention = self.encoder(out, return_attention=True)
        else:
            for layer in self.transformer:
                out = layer(out, domain=domain)
        return out, attention if return_attention else None

class InceptionFormer(PhenoFormer):
    def __init__(
            self,
            target_list,
            d_in=7,
            d_out=1,
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            n_layers=1,
            positional_encoding=True,
            elevation=False,
            latlon=False,
            T_pos_enc=1000,
            phases_as_input=None,
            norm_type="layernorm",
            use_nll=False,
            gated_attn=False,
            use_cross_attn=False,
            transformer_layer=TransformerEncoderLayer,
            residual=False,
            **kwargs
    ):
        super().__init__(
            target_list=target_list,
            d_in=d_in,
            d_out=d_out,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            positional_encoding=positional_encoding,
            elevation=elevation,
            latlon=latlon,
            T_pos_enc=T_pos_enc,
            phases_as_input=phases_as_input,
            norm_type=norm_type,
            use_nll=use_nll,
            gated_attn=gated_attn,
            use_cross_attn=use_cross_attn,
            residual=residual,
        )

        # Inception block
        self.inception_block = InceptionBlock(ni=d_in, nf=16, depth=4) # default as in paper: nf=32, depth=6, ks=40, bottleneck=True, residual=True

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Transformer layer for temporal encoding 
        self.transformer = nn.ModuleList(
            [
                transformer_layer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_type=norm_type,
                    gated_attn=gated_attn,
                )
                for _ in range(n_layers)
            ]
        )
    
    def forward(self, batch, return_attention=False, domain="source"):
        return super().forward(batch, return_attention, domain)
    
    def forward_residual(self, batch, return_attention=False, domain="source"):
        return super().forward_residual(batch, return_attention, domain)
    
    def forward_features(self, batch, return_attention=False, domain="source", return_features=False):  
        x = batch["climate"]
        b, t, _ = x.shape

        # Concatenation of static features (if any)
        if self.d_static > 0:
            static_data = []
            if self.elevation:
                static_data.append(batch["elevation_normalised"].unsqueeze(1))
            if self.latlon:
                static_data.append(batch["latlon_normalised"])
            if self.phases_as_input is not None:
                for p in self.phases_as_input:
                    static_data.append(batch["input_phases"][p].unsqueeze(1))
            static_data = torch.cat(static_data, 1)
            x = torch.cat([x, static_data.unsqueeze(1).repeat(1, t, 1)], dim=2)
            b, t, _ = x.shape

        # Run inceptionTime block
        x = x.permute(0, 2, 1)  # from [b, t, c] → [b, c, t]
        inception_out = self.inception_block(x)
        out = inception_out.permute(0, 2, 1)  # from [b, c, t] → [b, t, c]

        # Add positional encoding to encode the time information
        if self.positional_encoding is not None:
            positions = batch["doys"]
            if out.device != positions.device:
                print(f'----------- {out.device}, {positions.device}')
            out = out + self.positional_encoding(positions) # out.shape=torch.Size([16, 365, 128]), positions.shape=torch.Size([16, 365])

        # Prepend learnt tokens
        learnt_tokens = torch.cat([self.learnt_tokens[t] for t in self.target_list], 0)
        learnt_tokens = learnt_tokens.unsqueeze(0).repeat((b, 1, 1))

        if hasattr(self, "cross_attn"):
            learnt_tokens, _ = self.cross_attn(query=learnt_tokens, key=out, value=out)
            inception_out = learnt_tokens
        else:
            inception_out = inception_out.permute(0, 2, 1)

        # Apply transformer layer
        if isinstance(self.transformer[0], TransformerDecoderLayer):
            for layer in self.transformer:
                learnt_tokens = layer(learnt_tokens, out)
            out = learnt_tokens
        else:
            out = torch.cat([learnt_tokens, out], dim=1)
            if return_attention:
                if isinstance(self.transformer, nn.ModuleList):
                    attentions = []
                    for layer in self.transformer:
                        out, attention = layer(out, return_attention=True)
                        attentions.append(attention)
                    attention = torch.stack(
                        attentions, 0
                    )  # n_layer x B x n_head x target_sequence x source_sequence
                else:
                    out, attention = self.transformer(out, return_attention=True)
            else:
                for layer in self.transformer:
                    out = layer(out, domain=domain)

        if return_features:
            out = {"transformer_features": out, "features": inception_out}

        return out, attention if return_attention else None

class GRUFormer(PhenoFormer):
    def __init__(
            self,
            target_list,
            d_in=7,
            d_out=1,
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            n_layers=1,
            positional_encoding=True,
            elevation=False,
            latlon=False,
            T_pos_enc=1000,
            phases_as_input=None,
            norm_type="layernorm",
            use_nll=False,
            gated_attn=False,
            transformer_layer=TransformerEncoderLayer,
            use_cross_attn=False,
            residual=False,
            **kwargs
    ):
        super().__init__(
            target_list=target_list,
            d_in=d_in,
            d_out=d_out,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            positional_encoding=positional_encoding,
            elevation=elevation,
            latlon=latlon,
            T_pos_enc=T_pos_enc,
            phases_as_input=phases_as_input,
            norm_type=norm_type,
            use_nll=use_nll,
            gated_attn=gated_attn,
            use_cross_attn=use_cross_attn,
            residual=residual,
        )

        # GRU
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=d_model // 2, # for bidirectional
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Transformer layer for temporal encoding 
        self.transformer = nn.ModuleList(
            [
                transformer_layer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_type=norm_type,
                    gated_attn=gated_attn,
                )
                for _ in range(n_layers)
            ]
        )
    
    def forward(self, batch, return_attention=False, domain="source"):
        return super().forward(batch, return_attention, domain)
    
    def forward_residual(self, batch, return_attention=False, domain="source"):
        return super().forward_residual(batch, return_attention, domain)
    
    def forward_features(self, batch, return_attention=False, domain="source", return_features=False):
        x = batch["climate"]
        b, t, _ = x.shape

        # Concatenation of static features (if any)
        if self.d_static > 0:
            static_data = []
            if self.elevation:
                static_data.append(batch["elevation_normalised"].unsqueeze(1))
            if self.latlon:
                static_data.append(batch["latlon_normalised"])
            if self.phases_as_input is not None:
                for p in self.phases_as_input:
                    static_data.append(batch["input_phases"][p].unsqueeze(1))
            static_data = torch.cat(static_data, 1)
            x = torch.cat([x, static_data.unsqueeze(1).repeat(1, t, 1)], dim=2)
            b, t, _ = x.shape

        # Run GRU
        out, _ = self.gru(x)

        # Add positional encoding to encode the time information
        if self.positional_encoding is not None:
            positions = batch["doys"]
            if out.device != positions.device:
                print(f'----------- {out.device}, {positions.device}')
            out = out + self.positional_encoding(positions) # out.shape=torch.Size([16, 365, 128]), positions.shape=torch.Size([16, 365])

        # Prepend learnt tokens
        learnt_tokens = torch.cat([self.learnt_tokens[t] for t in self.target_list], 0)
        learnt_tokens = learnt_tokens.unsqueeze(0).repeat((b, 1, 1))

        if hasattr(self, "cross_attn"):
            learnt_tokens, _ = self.cross_attn(query=learnt_tokens, key=out, value=out)
            gru_out = learnt_tokens
        else:
            gru_out = out

        # Apply transformer layer
        if isinstance(self.transformer[0], TransformerDecoderLayer):
            for layer in self.transformer:
                learnt_tokens = layer(learnt_tokens, out)
            out = learnt_tokens
        else:
            out = torch.cat([learnt_tokens, out], dim=1)
            if return_attention:
                if isinstance(self.transformer, nn.ModuleList):
                    attentions = []
                    for layer in self.transformer:
                        out, attention = layer(out, return_attention=True)
                        attentions.append(attention)
                    attention = torch.stack(
                        attentions, 0
                    )  # n_layer x B x n_head x target_sequence x source_sequence
                else:
                    out, attention = self.transformer(out, return_attention=True)
            else:
                for layer in self.transformer:
                    out = layer(out, domain=domain)

        if return_features:
            out = {"transformer_features": out, "features": gru_out}

        return out, attention if return_attention else None
    
class PhenoFormerReconstruct(PhenoFormer):
    def __init__(
            self,
            target_list,
            d_in=7,
            d_out=1,
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            n_layers=1,
            positional_encoding=True,
            elevation=False,
            latlon=False,
            T_pos_enc=1000,
            phases_as_input=None,
            norm_type="layernorm",
            gated_attn=False,
            use_cross_attn=False,
            **kwargs
    ):
        super().__init__(
            target_list=target_list,
            d_in=d_in,
            d_out=d_out,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            positional_encoding=positional_encoding,
            elevation=elevation,
            latlon=latlon,
            T_pos_enc=T_pos_enc,
            phases_as_input=phases_as_input,
            norm_type=norm_type,
            gated_attn=gated_attn,
        )
        self.decoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    # norm_type=norm_type
                )
                for _ in range(n_layers)
            ],
            nn.Linear(d_model, d_in)
        )

    def forward(self, batch, return_attention=False, domain="source"):
        out, attention = self.forward_features(batch, return_attention, domain)
        
        # Decoding
        # Retrieve the embedding of each learnt token (each phenophase)
        task_embeddings = out[:, : self.n_task, :]  # batch x n_task x d_model
        # Linear decoder for each phenophase d_model -> 1 (or d_out)
        preds = self.linear_decoder(task_embeddings)
        # Split the output into the different phenophases
        predictions = {
            self.target_list[i]: chunk[:, i, :].squeeze(1)
            for i, chunk in enumerate(preds.chunk(self.n_task, dim=2))
        }

        output = {"predictions": predictions}

        if hasattr(self, 'linear_decoder_var'):
            vars = self.linear_decoder_var(task_embeddings)
            variances = {
                self.target_list[i]: chunk[:, i, :].squeeze(1)
                for i, chunk in enumerate(vars.chunk(self.n_task, dim=2))
            }
            output["variances"] = variances

        if return_attention:
            output["attention"] = attention
        
        # self reconstruction
        x_re = self.decoder(out[:, self.n_task:, :])
        output["reconstructed"] = x_re

        return output


class PositionalEncoder(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

if __name__=="__main__":
    model = PhenoFormer(
        target_list=["European_beech:leaf_unfolding",
                        "European_larch:needle_emergence",
                        "Common_spruce:needle_emergence",
                        "Hazel:leaf_unfolding"]
                        )
    
    T = 365
    d = 7 
    batch_size = 32

    dummy_climate = torch.rand(batch_size, T, d)
    doys = torch.arange(T).unsqueeze(0).repeat(batch_size, 1)

    out = model({"climate": dummy_climate, "doys": doys})
    print(out)