from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
import numpy.typing as npt
import math
import torch
import torch.nn as nn
import os
import typing
import matplotlib.pyplot as plt


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


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,
    ):
        super().__init__()
        self.std = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

        self.init_weight()
    
    def init_weight(self):
        torch.nn.init.trunc_normal_(
            self.W1,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3,
        )
        torch.nn.init.trunc_normal_(
            self.W2,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3
        )
        torch.nn.init.trunc_normal_(
            self.W3,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3
        )

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        first_term = self.SiLU(torch.matmul(x, self.W1.T))
        second_term = torch.matmul(x, self.W3.T)
        return torch.matmul(torch.mul(first_term, second_term), self.W2.T)


class ROPE(nn.Module):
    def __init__(
        self,
        theta:float,
        d_k:int,
        max_seq_len:int,
        device=None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        # rotation matrix
        i = torch.arange(0, d_k//2, dtype=torch.float32)
        frequencies = theta ** (-2*i/d_k)
        positions = torch.arange(0, max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, frequencies) # shape: (max_seq_len, d_k // 2)

        # get cos and sin
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _rotate_half(self, x):
        """
        Rotate the last dimension of x by swapping and negating pairs of elements
        For RoPE, we rotate pairs of dimensions: (x1, x2) -> (-x2, x1)
        """
        # split into halves
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # interleave
        rotated = torch.stack((-x2, x1), dim=-1)  # shape (..., d_k // 2, 2)
        return rotated.flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Apply RoPE: x * cos + rotate(half) * sin
        rotated_x = self._rotate_half(x)
        return x * cos + rotated_x * sin


def softmax(x: torch.Tensor, i: int):
    # Subtract maximum for numerical stability. keepdim=True for broadcasting
    max_val = torch.max(x, dim=i, keepdim=True).values
    normalized = x - max_val
    # Apply exponent
    exp_vals = torch.exp(normalized)
    sum_exp = torch.sum(exp_vals, dim=i, keepdim=True)
    return exp_vals / sum_exp


def SDPA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Get the scaling factor
    d_k = Q.size(-1)

    # Calculate raw attention scores Q * K^T / sqrt(d_k)
    # Use transpose(-2, -1) to handle batched inputs safely
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply masking
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax
    softmax_output = softmax(scores, -1)
    return torch.matmul(softmax_output, V)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None,
                 rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.rope:
            if not theta or not max_seq_len:
                raise ValueError("theta and max_seq_len must be provided when applying RoPE")
            self.rope_emb = ROPE(theta, self.d_k, max_seq_len)

    def forward(self, x, token_positions: torch.Tensor = None):
        # create a causal mask during forward mask dynamically based on input seq length
        B, S, _ = x.size()
        causal_mask = torch.tril(torch.ones((S, S), dtype=torch.bool))

        # linear projection: x @ W. shape: (B, S, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # split heads
        # view as (B, S, num_heads, d_k), then transpose to (B, H, S, d_k)
        Q = Q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope:
            # default positions
            if token_positions is None:
                token_positions = torch.arange(S, device=x.device)
            Q = self.rope_emb(Q, token_positions)
            K = self.rope_emb(K, token_positions)

        # SDPA output shape: (B, num_heads, S, d_k)
        attention_output = SDPA(Q, K, V, mask=causal_mask)

        # concat head
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_O(attention_output)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        # initialize layers
        self.layer_norm_1 = RMSNorm(d_model=d_model)
        self.layer_norm_2 = RMSNorm(d_model=d_model)
        self.multihead_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=True,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x, token_positions: torch.Tensor = None):
        # RMSNorm -> Multi-head self attention -> residual connection
        layer_norm_output = self.layer_norm_1(x)
        multihead_attention_output = self.multihead_attention(layer_norm_output, token_positions=token_positions)
        residual_connection_output = multihead_attention_output + x

        # Position wise feedforward network: RMSNorm -> SwiGLU -> residual connection
        layer_norm_output_2 = self.layer_norm_2(residual_connection_output)
        feedforward_network_output = self.swiglu(layer_norm_output_2)
        return feedforward_network_output + residual_connection_output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.num_layers = num_layers
        # initialize layers
        # token embedding
        self.emb = Embedding(vocab_size, d_model)

        # transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)
            ]
        )
        # last layer norm
        self.rmsnorm = RMSNorm(d_model=d_model)
        # linear projection
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x, token_positions: torch.Tensor = None):
        B, S = x.shape
        # Generate token positions for RoPE
        # unsqueeze(0): add a new dimension at index 0 so [4] -> [1,4]
        # expand(B, -1): expand the first dimension to size B, -1 tells pytorch to leave that dimension as is
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        # embed (B, S) -> (B, S, d_model)
        token_emb = self.emb(x)
        # apply transformer blocks
        for layer_i in self.transformer_blocks:
            token_emb = layer_i(token_emb, token_positions)
        # apply layer norm
        layernorm_output = self.rmsnorm(token_emb)
        # linear projection to vocab size
        logits = self.linear(layernorm_output)
        # apply softmax
        # return softmax(linear_output, i=-1)
        return logits


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs: (B, C). targets: (B)
    max_val = torch.max(inputs, dim=-1, keepdim=True).values
    normalized_inputs = inputs - max_val
    # true-class logits: (B, 1)
    true_class_logit = torch.gather(normalized_inputs, dim=1, index=targets.unsqueeze(1))
    # get the sum of logit
    logit_sum = torch.log(torch.sum(torch.exp(normalized_inputs), keepdim=True, dim=1))
    total_loss = logit_sum - true_class_logit
    return torch.mean(total_loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)   # Get state associated with p
                grad = p.grad.data  # Get the gradient of loss w.r.t. p
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in place
                state["t"] = t + 1  # Increment iteration number


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # A dictionary containing default values for the optimizer's hyperparameters
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # param update
                grad = p.grad.data  # Get gradient of loss w.r.t p
                # get state
                state = self.state[p]
                # initialize state on first use
                t = state.get("t", 0)  # Get iteration number
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t += 1
                # compute adjusted lr for this iteration
                lr_adjusted = lr * (np.sqrt(1-beta2**t) / (1-beta1**t))
                # apply weight decay, distinct from parameter update
                p.data = (1 - lr * weight_decay) * p.data
                # Update first and second moment
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * grad**2
                # Update weight
                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)
                # Update state: iteration number, first and second moment
                state["t"] = t
                state["m"] = m
                state["v"] = v


def cosine_annealing_scheduler(t: int, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    if t < 0:
        raise ValueError(f"Invalid iteration: {t}")
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif t >= warmup_steps and t <= total_steps:
        return lr_min + 1/2 * (1 + np.cos((t - warmup_steps) / (total_steps - warmup_steps) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    # 1. Collect active gradients
    grads = [p.grad.data for p in parameters if p.grad is not None]
    if not grads:
        return
    # 2. Compute l2 norm
    total_norm = torch.sqrt(sum(torch.linalg.norm(g) ** 2 for g in grads))
    # 3. Scale gradients
    if total_norm > max_l2_norm:
        scale_factor = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale_factor)


def load_data(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> (torch.Tensor, torch.Tensor):
    total_len = x.shape[0]
    data, label = [], []
    # sample starting indices
    start_indices = np.random.randint(
        0,
        total_len - context_length,
        size=batch_size,
    )
    for i in start_indices:
        input = x[i : i+context_length]
        # target is shifted one position to the right
        target = x[i+1 : i+context_length+1]
        data.append(input)
        label.append(target)
       
    inputs =  torch.tensor(np.array(data), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(label), dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    # Merge into a single nested dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)
    

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    iteration = ckpt["iteration"]
    return iteration
    


if __name__ == "__main__":
    lr_max = 1e-3
    lr_min = 1e-6
    warmup_steps = 10_000
    total_steps = 100_000
    lr_list = []
    for t in range(total_steps):
        lr = cosine_annealing_scheduler(t, lr_max, lr_min, warmup_steps, total_steps)
        lr_list.append(lr)
    plt.figure(figsize=(10,6))
    plt.plot(range(total_steps), lr_list)
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.grid(True)
    plt.show()
    # beta1 = 0.9
    # beta2 = 0.99
    # lr_list = []
    # lr_init = 1e-3
    # total_steps = 10000
    # for t in range(1, total_steps, 1):
    #     lr_adj = lr_init * (np.sqrt(1-beta2**t) / (1-beta1**t))
    #     lr_list.append(lr_adj)
    # plt.figure(figsize=(10,6))
    # for lr in lr_list:
    #     plt.plot(range(1, total_steps, 1), lr_list)
    # plt.xlabel("Training step")
    # plt.ylabel("Adjusted learning rate")
    # plt.legend()
    # plt.grid(True)
    # plt.show()



# if __name__ == "__main__":
#     initial_weights = torch.nn.Parameter(5 * torch.randn((10,10)))
#     learning_rates = [1]
#     losses_by_lr = {}
#     for lr in learning_rates:
#         # Reset to identical starting weights
#         weights = torch.nn.Parameter(initial_weights.clone())
#         opt = AdamW([weights], lr=lr, beta1=0.9, beta2=0.999, eps=1e-4, weight_decay=1e-2)
#         losses = []
#         for t in range(100):
#             opt.zero_grad()
#             loss = (weights ** 2).mean()
#             # Save scalar loss
#             losses.append(loss.item())
#             loss.backward() # Run backward pass
#             opt.step()  # Run optimizer step
#         losses_by_lr[lr] = losses
#     # plot
#     plt.figure(figsize=(10, 6))
#     for lr, losses in losses_by_lr.items():
#         plt.plot(range(100), losses, label=f"lr={lr}")
#     plt.xlabel("Training step")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
