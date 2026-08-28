"""Pre-LayerNorm transformer (character-level), with full intermediate tracing.

Conventions (binding, see README.md):
  * GELU = tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    realized as nn.GELU(approximate="tanh") — never the erf variant.
  * LayerNorm: eps = 1e-5, biased variance (no Bessel correction), weight AND bias.
  * Every nn.Linear has bias=True. No exceptions.
  * No weight tying: lm_head.weight is independent of wte.weight.
  * Learned absolute position embeddings (wpe); position = index in the context
    window (0..T-1). No RoPE, no ALiBi.
  * Attention is written out explicitly; the Q @ K^T scaling, the causal mask,
    the softmax and the value-weighted sum are each separate, visible steps — no
    fused kernel may hide any of the intermediates.
  * Single fused QKV projection nn.Linear(n_embd, 3 * n_embd); the output is split
    in the order [Q | K | V]. Within a block, head h occupies columns
    h*head_dim .. h*head_dim + head_dim - 1.
  * scores = (Q @ K^T) / sqrt(head_dim) (here sqrt(32)); causal mask for s > t.
  * Softmax is numerically stable: row max subtracted, then exp, then divide by sum.
  * No dropout anywhere (p would be 0.0); no dropout module is built at all.
  * Everything is computed in float32; no autocast, no bf16, no TF32 matmul.

Tensor-shape comments are given in dimension order and are part of the contract.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model hyperparameters.

    Fields (dimension order n_layer, n_head, n_embd, head_dim is derived,
    block_size, vocab_size, mlp_hidden):
      n_layer:    int — number of transformer blocks (4)
      n_head:     int — attention heads per block (4)
      n_embd:     int — model width C (128); head_dim = n_embd / n_head (32)
      block_size: int — max context window T (64)
      vocab_size: int — vocabulary V, from data/meta.json (65)
      mlp_hidden: int — MLP hidden width (512 = 4 * n_embd)
    """

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    vocab_size: int = 65
    mlp_hidden: int = 512


class LayerNorm(nn.Module):
    """Own LayerNorm so the arithmetic is visible.

    Input:  (B, T, C)
    Output: (B, T, C)
    Mean and variance over the last axis C only, biased variance (no Bessel
    correction), eps = 1e-5, with learnable weight (C,) and bias (C,).
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        # weight: (C,); bias: (C,)
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))  # (C,)
        self.bias = nn.Parameter(torch.zeros(n_embd))  # (C,)

    def forward(self, x):
        # x: (B, T, C)
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # (B, T, 1)
        xhat = (x - mean) / torch.sqrt(var + self.eps)  # (B, T, C)
        return self.weight * xhat + self.bias  # (B, T, C)


class CausalSelfAttention(nn.Module):
    """Explicit causal multi-head self-attention, no fused kernels.

    Input:  x (B, T, C)
    Output: (B, T, C)
    One single projection nn.Linear(C, 3C, bias=True) to [Q | K | V]:
    Q = columns 0..127, K = 128..255, V = 256..383. Within a block head h owns
    columns h*head_dim .. h*head_dim + head_dim - 1 (head_dim = C / n_head = 32).
    scores = (Q @ K^T) / sqrt(head_dim), causally masked for s > t; stable softmax.
    """

    def __init__(self, cfg: GPTConfig):
        # c_attn.weight: (3C, C); c_attn.bias: (3C,); c_proj.weight: (C, C); c_proj.bias: (C,)
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head  # 32
        # c_attn: (C) -> (3C) with bias; Q = columns 0..127, K = 128..255, V = 256..383.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)

    def forward(self, x, trace, l):
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)

        # (B, T, n_head*head_dim) -> (B, n_head, T, head_dim):
        def reshape_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        qh = reshape_heads(q)  # (B, n_head, T, head_dim)
        kh = reshape_heads(k)  # (B, n_head, T, head_dim)
        vh = reshape_heads(v)  # (B, n_head, T, head_dim)

        # scores = Q @ K^T / sqrt(head_dim): (B, n_head, T, T);
        # row t = query position, column s = key position.
        scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(32)
        scores_scaled = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B, n_head, T, T)
        # Causal mask: entry (t, s) is masked for s > t. Internally -inf; the
        # dump replaces it with -1.0e30.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores_masked = scores_scaled.masked_fill(mask[None, None, :, :], float("-inf"))  # (B, n_head, T, T)
        # Numerically stable softmax: row max subtracted, then exp, then divide by sum.
        probs = F.softmax(scores_masked, dim=-1)  # (B, n_head, T, T); each row sums to 1
        head_out = torch.matmul(probs, vh)  # (B, n_head, T, head_dim)
        # Concatenate heads h=0..n_head-1 side by side:
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        y = head_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)  # (B, T, C), includes the projection bias

        if trace is not None:
            for h in range(self.n_head):
                trace[f"L{l}_11_q_h{h}"] = qh[0, h].detach().cpu()  # (T, head_dim)
                trace[f"L{l}_12_k_h{h}"] = kh[0, h].detach().cpu()  # (T, head_dim)
                trace[f"L{l}_13_v_h{h}"] = vh[0, h].detach().cpu()  # (T, head_dim)
                trace[f"L{l}_14_scores_scaled_h{h}"] = scores_scaled[0, h].detach().cpu()  # (T, T), UNmasked
                trace[f"L{l}_15_scores_masked_h{h}"] = scores_masked[0, h].detach().cpu()  # (T, T)
                trace[f"L{l}_16_attn_probs_h{h}"] = probs[0, h].detach().cpu()  # (T, T)
                trace[f"L{l}_17_head_out_h{h}"] = head_out[0, h].detach().cpu()  # (T, head_dim)
            trace[f"L{l}_18_attn_concat"] = y[0].detach().cpu()  # (T, C), heads h=0..n_head-1
        return out  # (B, T, C)


class MLP(nn.Module):
    """Two-layer feed-forward block with tanh-GELU.

    Input:  (B, T, C)
    Hidden: (B, T, mlp_hidden) with mlp_hidden = 4 * C = 512
    Output: (B, T, C)
    c_fc: weight (512, 128), bias (512,); c_proj: weight (128, 512), bias (128,)
    """

    def __init__(self, cfg: GPTConfig):
        # c_fc.weight: (512, C); c_fc.bias: (512,); c_proj.weight: (C, 512); c_proj.bias: (C,)
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=True)
        self.gelu = nn.GELU(approximate="tanh")  # tanh approximation, not erf
        self.c_proj = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=True)

    def forward(self, x, trace, l):
        # x: (B, T, C)
        x1 = self.c_fc(x)  # (B, T, mlp_hidden) incl. bias
        x2 = self.gelu(x1)  # (B, T, mlp_hidden)
        out = self.c_proj(x2)  # (B, T, C) incl. bias
        if trace is not None:
            trace[f"L{l}_31_fc"] = x1[0].detach().cpu()  # (T, mlp_hidden)
            trace[f"L{l}_32_gelu"] = x2[0].detach().cpu()  # (T, mlp_hidden)
            trace[f"L{l}_33_mlp_proj"] = out[0].detach().cpu()  # (T, C)
        return out  # (B, T, C)


class Block(nn.Module):
    """One transformer block: x = x + attn(ln1(x)); then x = x + mlp(ln2(x)).

    Input:  (B, T, C)
    Output: (B, T, C)
    """

    def __init__(self, cfg: GPTConfig):
        # ln1/ln2 weight,bias: (C,); attn: see CausalSelfAttention; mlp: see MLP.
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, trace, l):
        # x: (B, T, C)
        ln1_out = self.ln1(x)  # (B, T, C)
        if trace is not None:
            trace[f"L{l}_10_ln1"] = ln1_out[0].detach().cpu()  # (T, C)
        attn_out = self.attn(ln1_out, trace, l)  # (B, T, C)
        if trace is not None:
            trace[f"L{l}_19_attn_proj"] = attn_out[0].detach().cpu()  # (T, C)
        x = x + attn_out  # (B, T, C), residual after attention
        if trace is not None:
            trace[f"L{l}_20_resid_post_attn"] = x[0].detach().cpu()  # (T, C)
        ln2_out = self.ln2(x)  # (B, T, C)
        if trace is not None:
            trace[f"L{l}_30_ln2"] = ln2_out[0].detach().cpu()  # (T, C)
        mlp_out = self.mlp(ln2_out, trace, l)  # (B, T, C)
        x = x + mlp_out  # (B, T, C), residual after MLP
        if trace is not None:
            trace[f"L{l}_34_resid_post_mlp"] = x[0].detach().cpu()  # (T, C)
        return x  # (B, T, C)


class GPT(nn.Module):
    """Pre-LayerNorm transformer language model.

    forward:
        Input:  idx (B, T) int64 token ids (B = 1 for the reference dump);
                targets (B, T) int64 or None; trace dict or None.
        Output: logits (B, T, V); loss scalar (B,) if targets given, else None.
        When trace is not None, every intermediate is stored in it, keyed exactly
        as in the trace-key table (without ".csv"), as a detached CPU tensor
        WITHOUT the batch dimension — (T, 1), (T, C), (T, T) or (1, V). When
        trace is None the forward incurs no extra work. There is exactly ONE
        forward path; there is no separate "trace forward".

    generate:
        Input:  idx (B, T) int64; max_new_tokens int; temperature float;
                generator torch.Generator or None.
        Output: (B, T + max_new_tokens) int64 — the full token sequence.
    """

    def __init__(self, cfg: GPTConfig):
        # wte.weight: (V, C); wpe.weight: (block_size, C);
        # ln_f.weight/bias: (C,); lm_head.weight: (V, C); lm_head.bias: (V,)
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),  # (V, C)
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),  # (block_size, C)
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd),
            )
        )
        # Independent output head: NO weight tying to wte.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # module parameters keep their shapes; only values are initialized.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, trace=None):
        # idx: (B, T) int64; targets: (B, T) int64 or None; trace: dict or None.
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size {self.cfg.block_size}"
        toks = self.transformer.wte(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)  # (T,) — position = index in window
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        x = toks + pos_emb  # (B, T, C)

        if trace is not None:
            trace["00_tokens"] = idx[0].unsqueeze(1).detach().cpu()  # (T, 1) int
            trace["01_tok_emb"] = toks[0].detach().cpu()  # (T, C)
            trace["02_pos_emb"] = pos_emb.detach().cpu()  # (T, C)
            trace["03_x_input"] = x[0].detach().cpu()  # (T, C)

        for l, block in enumerate(self.transformer.h):
            x = block(x, trace, l)  # (B, T, C)

        x = self.transformer.ln_f(x)  # (B, T, C)
        if trace is not None:
            trace["90_ln_f"] = x[0].detach().cpu()  # (T, C)
        logits = self.lm_head(x)  # (B, T, V)
        if trace is not None:
            trace["91_logits"] = logits[0].detach().cpu()  # (T, V)
            last = logits[0, -1, :]  # (V,)
            trace["92_logits_last"] = last.unsqueeze(0).detach().cpu()  # (1, V)
            last_probs = F.softmax(last, dim=-1)  # (V,), temperature 1.0
            trace["93_probs_last_temp1"] = last_probs.unsqueeze(0).detach().cpu()  # (1, V)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, generator=None):
        # idx: (B, T) int64; returns (B, T + max_new_tokens) int64.
        # Context is clipped to the last block_size tokens at every step; logits of
        # the last position are divided by temperature, softmaxed, and sampled
        # with torch.multinomial (optional generator for reproducibility).
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]  # (B, <= block_size)
            logits, _ = self(context)  # (B, T_cur, V)
            last = logits[:, -1, :] / temperature  # (B, V)
            probs = F.softmax(last, dim=-1)  # (B, V)
            if generator is None:
                next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_tok = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
            idx = torch.cat((idx, next_tok), dim=1)  # (B, T + 1)
        return idx
