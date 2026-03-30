"""

Normalisation layers for domain adaptations.

"""

import torch
import torch.nn as nn
import torch.nn.init as init

class DomainAgnosticLayerNorm(nn.Module):
    """
    paper: A Domain Agnostic Normalization Layer for Unsupervised Adversarial Domain Adaptation
    """
    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, dim=-1, device=None, dtype=None,
                 adaptive=False):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.weight = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, **factory_kwargs))
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        # for adaptive norm
        self.adaptive = adaptive
        self.k = 0.1
        print(f">> use adaptive norm: {adaptive}")

        # Fixed stats to use for all non-source domains
        self.register_buffer("source_mean", None)
        self.register_buffer("source_std", None)

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x, domain='source'):
        if domain == 'source':
            mean = x.mean(dim=self.dim, keepdim=True)
            std = x.std(dim=self.dim, keepdim=True)
            if self.source_mean is None:
                self.source_mean = mean.detach()
                self.source_std = std.detach()
            else:
                # Save running stats during source training
                # update running stats
                self.source_mean = (1 - self.momentum) * self.source_mean + self.momentum * mean.detach()
                self.source_std  = (1 - self.momentum) * self.source_std  + self.momentum * std.detach()
        else:
            # handle sanity check with validation data
            if self.source_mean is None:
                mean = x.mean(dim=self.dim, keepdim=True)
                std = x.std(dim=self.dim, keepdim=True)
            else:
                if self.source_mean.shape[0] > 1:
                    # make batch mean
                    self.source_mean = self.source_mean.mean(dim=0, keepdim=True)
                    self.source_std = self.source_std.mean(dim=0, keepdim=True)

                # Use source stats for target/unseen
                mean = self.source_mean
                std = self.source_std
        
        if self.dim == -1 and x.shape[0] != mean.shape[0]:
            mean = mean[:x.shape[0]]
            std = std[:x.shape[0]]
            
        out = (x - mean) / (std + self.eps)

        if self.adaptive:
            return self.weight * (1 - self.k * out ) * out + self.bias
        
        return self.weight * out + self.bias

class DomainAgnosticRMSNorm(nn.Module):
    """
    Domain-agnostic RMSNorm:
    - During source training: compute RMS stats and update running buffers
    - During target inference: normalize using stored source running stats

    Similar spirit to "Domain Agnostic Normalization Layer" but adapted for RMSNorm.
    """

    def __init__(self, normalized_shape, eps=1e-6, momentum=0.1, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
        self.eps = eps
        self.momentum = momentum

        # store scalar RMS (or vector RMS) statistics
        self.register_buffer("source_rms", None)

    def forward(self, x, domain="source"):
        """
        x shape: [B, T, D] or [B, D]
        RMSNorm normally computes stats along last dim (-1).
        """

        if domain == "source":
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            if self.source_rms is None:
                self.source_rms = rms.detach()
            else:
                # update running RMS
                self.source_rms = (1 - self.momentum) * self.source_rms + self.momentum * rms.detach()
        else:
            # handle sanity check with validation data
            if self.source_rms is None:
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            else:
                if self.source_rms.shape[0] > 1:
                    self.source_rms = self.source_rms.mean(dim=0, keepdim=True)
                rms = self.source_rms

        # handle batch mismatch (e.g. last batch smaller)
        if rms.shape[0] != x.shape[0]:
            rms = rms[: x.shape[0]]

        out = x / rms
        return out * self.weight


class AdaptiveBatchNorm1d(nn.Module):
    """
    Adaptive Batch Normalization (AdaBN)
    
    Works like BatchNorm1d, but during inference on target domain,
    it uses target-domain population statistics instead of source stats.
    
    Input expected: [N, C, L] (same as nn.BatchNorm1d).

    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # source domain running stats
        self.register_buffer("running_mean_source", torch.zeros(num_features))
        self.register_buffer("running_var_source", torch.ones(num_features))

        # target domain running stats (computed during adaptation stage)
        self.register_buffer("running_mean_target", torch.zeros(num_features))
        self.register_buffer("running_var_target", torch.ones(num_features))

        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
        self.running_mean_source.zero_()
        self.running_var_source.fill_(1)
        self.running_mean_target.zero_()
        self.running_var_target.fill_(1)
        self.num_batches_tracked.zero_()

    def forward(self, x, domain="source"):
        """
        x: [N, C, L]
        domain: "source" or "target"
        mode: "train" or "test"
        
        - During source training: update source running stats.
        - During target adaptation: update target running stats.
        - During target testing: use target running stats.
        """
        assert x.dim() == 3, "Expected input shape [N, C, L]"

        if self.training:
            # compute batch statistics
            mean = x.mean(dim=(0, 2))   # [C]
            var = x.var(dim=(0, 2), unbiased=False)  # [C]

            if domain == "source":
                self.running_mean_source = (1 - self.momentum) * self.running_mean_source + self.momentum * mean.detach()
                self.running_var_source  = (1 - self.momentum) * self.running_var_source  + self.momentum * var.detach()
            else:
                self.running_mean_target = (1 - self.momentum) * self.running_mean_target + self.momentum * mean.detach()
                self.running_var_target  = (1 - self.momentum) * self.running_var_target  + self.momentum * var.detach()

            mean_used = mean
            var_used = var

        else:  # mode == "test"
            if domain == "source":
                mean_used = self.running_mean_source
                var_used = self.running_var_source
            else:
                mean_used = self.running_mean_target
                var_used = self.running_var_target


        # reshape for broadcasting: [C] -> [1, C, 1]
        mean_used = mean_used.view(1, -1, 1)
        var_used = var_used.view(1, -1, 1)

        x_hat = (x - mean_used) / torch.sqrt(var_used + self.eps)

        if self.affine:
            w = self.weight.view(1, -1, 1)
            b = self.bias.view(1, -1, 1)
            x_hat = w * x_hat + b

        return x_hat

