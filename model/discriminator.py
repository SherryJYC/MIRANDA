"""
Discriminator model.

"""
import torch
from torch import nn
from model.transformer_pytorch import TransformerDecoderLayer

def getDiscriminator(d_model, n_task, 
                     input_dims,
                     hidden_dims=64, 
                     output_dims=2,
                     nhead=8, 
                     n_layers=1,
                     use_softmax=False,
                     discriminator_type="mlp",
                     rank_mask=False,
                     avg_pool=False,
                     cross_attn=False):
    if discriminator_type == "mlp":
        return Discriminator(input_dims=input_dims,
                             hidden_dims=hidden_dims,
                             output_dims=output_dims,
                             use_softmax=use_softmax,
                             rank_mask=rank_mask,
                             avg_pool=avg_pool,
                             cross_attn=cross_attn)
    elif discriminator_type == "transformer":
        return DiscriminatorTrans(d_model=d_model,
                                  hidden_dims=hidden_dims,
                                  output_dims=output_dims,
                                  nhead=nhead, n_layers=n_layers,
                                  use_softmax=use_softmax)
    else:
        raise ValueError(f"Unknown discriminator type {discriminator_type}!")
    

class Discriminator(nn.Module):
    """Discriminator model."""

    def __init__(self, input_dims=64, hidden_dims=64, output_dims=2,
                 use_softmax=False, rank_mask=False, avg_pool=False,
                 cross_attn=False):
        super(Discriminator, self).__init__()

        self.restored = False
        self.use_softmax = use_softmax

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
        )

        self.softmax = nn.LogSoftmax(1)
        self.rank_mask = rank_mask
        self.avg_pool = avg_pool
        self.cross_attn = cross_attn
        if cross_attn:
            self.embedding = nn.Parameter(torch.randn(1, 1, input_dims))
            self.cross_attention = nn.MultiheadAttention(embed_dim=input_dims, num_heads=8, batch_first=True)
        print(f">> discriminator: rank_mask={rank_mask}, avg_pool={avg_pool}, cross_attn={cross_attn}")

    def forward(self, input):
        if self.avg_pool:
            input = input.mean(dim=1, keepdim=False) # [b, c]
        elif self.cross_attn:
            x = self.embedding.repeat(input.shape[0], 1, 1) # [B, 1, 1]
            input = self.cross_attention(query=x, key=input, value=input)[0]
        elif not self.rank_mask:
            input = input.reshape((input.shape[0], -1)) # [b, n_task*d_dim]
        out = self.layer(input)
        if self.use_softmax:
            out = self.softmax(out)
        return out
    
class DiscriminatorTrans(nn.Module):
    """Discriminator model with transformer layers."""

    def __init__(self, d_model=64, hidden_dims=64, output_dims=2,
                 nhead=8, n_layers=1,
                 use_softmax=False):
        super(DiscriminatorTrans, self).__init__()
        print(f"n_layers = {n_layers}")
        self.restored = False
        self.use_softmax = use_softmax
        
        self.embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=hidden_dims,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(d_model, output_dims)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, input):
        x = self.embedding.repeat(input.shape[0], 1, 1) # [B, 1, 1]
        for layer in self.layers:
            x = layer(x, input)
        out = self.linear(x).squeeze()
        
        if self.use_softmax:
            out = self.softmax(out).squeeze()
        return out
