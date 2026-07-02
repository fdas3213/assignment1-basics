import numpy as np
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        self.stddev = np.sqrt(2.0 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

    def init_weight(self):
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=self.stddev,
            a=-3.0*self.stddev,
            b=-3.0*self.stddev,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W.T)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings:int,
        embedding_dim:int,
        device:torch.device|None=None,
        dtype:torch.dtype|None=None,
    ):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.init_emb()
    
    def init_emb(self):
        torch.nn.init.trunc_normal_(
            self.emb,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb[x]


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * g
    """
    def __init__(
        self,
        d_model:int,
        eps:float=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        
        # Mean of square over d_model dimension
        first_term = 1 / self.d_model * torch.sum(x ** 2, dim=-1, keepdim=True)
        denom = torch.sqrt(first_term + self.eps)
        result = (x / denom) * self.g
        return result.to(in_type)




# if __name__ == "__main__":
#     linear_layer = Linear(20, 40)
#     x = torch.rand(20)
#     y = linear_layer(x)
#     # print(y.size())
#     emb_layer = Embedding(40, 16)
#     ind = torch.tensor([2, 9, 34])
#     print(emb_layer(ind))