# %% [markdown]
# # 151 Transformer code -- encoder + decoder
# 
# Minimal **PyTorch** Transformer with **shape tracking** at every step.
# 
# Companion intuition: 0850_transformer.py | Positional encoding: 0880_positional_encoding.py
# 
# > **Import note:** 0930_machine_translation.py / 0930_machine_translation.ipynb import **0910_transformer_code.py**, not this notebook.
# > This notebook is a shape tour; some helpers used by 152 (last_attn_weights, configurable vocab sizes, etc.) live only in the .py file.
# 

# %%
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
B = 2          # batch
S = 5          # source length
T = 4          # target length
D = 32         # d_model
H = 4          # n_heads
Dk = D // H    # per-head dim
D_FF = 64      # d_ff (avoid shadowing torch.nn.functional as F)
V_SRC = 50
V_TGT = 60
N_ENC = 2
N_DEC = 2
PAD_ID = 0

assert D % H == 0
torch.manual_seed(0)

def show(name: str, x: torch.Tensor) -> torch.Tensor:
    """Print tensor shape then return x unchanged."""
    print(f"  {name:28s} {tuple(x.shape)}")
    return x

print(f"Config: B={B}, S={S}, T={T}, D={D}, H={H}, Dk={Dk}, D_FF={D_FF}")


# %% [markdown]
# ## Positional encoding
# 
# Sinusoidal PE is **added** to token embeddings: `(B, L, D) + (1, L, D) -> (B, L, D)`.
# 

# %%
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # (L, D)
        pos = torch.arange(max_len).unsqueeze(1)    # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


# %% [markdown]
# ## Multi-head attention
# 
# Shapes inside one attention call:
# 
# 1. `query/key/value`: `(B, L, D)`
# 2. project + split heads: `(B, H, L, Dk)`
# 3. scores `Q K^T / sqrt(Dk)`: `(B, H, Lq, Lk)`
# 4. softmax weights: `(B, H, Lq, Lk)`
# 5. `weights @ V`: `(B, H, Lq, Dk)`
# 6. merge + output proj: `(B, Lq, D)`
# 

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.H = n_heads
        self.Dk = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dk)
        B, L, _ = x.shape
        return x.view(B, L, self.H, self.Dk).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dk) -> (B, L, D)
        B, H, L, Dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dk)

    def forward(
        self,
        query: torch.Tensor,   # (B, Lq, D)
        key: torch.Tensor,     # (B, Lk, D)
        value: torch.Tensor,   # (B, Lk, D)
        attn_mask: torch.Tensor | None = None,
        track: bool = False,
        tag: str = "attn",
    ) -> torch.Tensor:
        if track:
            print(f"    [{tag}] MultiHeadAttention")
            show(f"{tag} query", query)
            show(f"{tag} key", key)
            show(f"{tag} value", value)

        Q = self._split_heads(self.W_q(query))   # (B, H, Lq, Dk)
        K = self._split_heads(self.W_k(key))     # (B, H, Lk, Dk)
        V = self._split_heads(self.W_v(value))   # (B, H, Lk, Dk)
        if track:
            show(f"{tag} Q heads", Q)
            show(f"{tag} K heads", K)
            show(f"{tag} V heads", V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.Dk)  # (B, H, Lq, Lk)
        if track:
            show(f"{tag} scores QK^T/sqrt", scores)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)      # (B, H, Lq, Lk)
        if track:
            show(f"{tag} softmax weights", weights)

        ctx = torch.matmul(weights, V)           # (B, H, Lq, Dk)
        if track:
            show(f"{tag} weighted V", ctx)

        out = self.W_o(self._merge_heads(ctx))   # (B, Lq, D)
        if track:
            show(f"{tag} out (merged)", out)
        return out


# %% [markdown]
# ## Feed-forward network
# 
# `(B, L, D) -> (B, L, D_FF) -> (B, L, D)` applied independently at each position.
# 

# %%
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, track: bool = False, tag: str = "ffn") -> torch.Tensor:
        if track:
            show(f"{tag} in", x)
        y = self.net[0](x)
        if track:
            show(f"{tag} after Linear->F", y)
        y = self.net[1](y)
        y = self.net[2](y)
        if track:
            show(f"{tag} out", y)
        return y


# %% [markdown]
# ## Encoder
# 
# Each layer: **self-attention** (bidirectional, pad-masked) + **FFN**, with residuals and LayerNorm.
# 
# Final encoder output is the **memory** tensor `(B, S, D)` consumed by the decoder.
# 

# %%
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None, track: bool = False):
        # x: (B, S, D)
        if track:
            print("  -- EncoderLayer --")
            show("enc layer in", x)

        h = self.norm1(x)
        a = self.self_attn(h, h, h, attn_mask=src_mask, track=track, tag="enc-self")
        x = x + a                              # (B, S, D)
        if track:
            show("after enc self-attn+res", x)

        h = self.norm2(x)
        x = x + self.ffn(h, track=track, tag="enc-ffn")
        if track:
            show("enc layer out", x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pe = SinusoidalPE(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_ids, src_mask=None, track: bool = False):
        # src_ids: (B, S)
        if track:
            print("\n=== ENCODER ===")
            show("src_ids", src_ids)

        x = self.embed(src_ids) * math.sqrt(D)  # (B, S, D)
        if track:
            show("src embed * sqrt(D)", x)
        x = self.pe(x)                          # (B, S, D)
        if track:
            show("src + PE", x)

        for i, layer in enumerate(self.layers):
            if track:
                print(f"  [encoder layer {i}]")
            x = layer(x, src_mask, track=track)

        x = self.norm(x)                        # (B, S, D)  = memory
        if track:
            show("encoder memory", x)
        return x


# %% [markdown]
# ## Decoder
# 
# Each layer has three sub-blocks:
# 
# 1. **Masked self-attention** on the target `(B, T, D)` -- causal, no future
# 2. **Cross-attention**: queries from decoder, keys/values from encoder memory `(B, S, D)`
# 3. **FFN**
# 
# Then a linear head maps `(B, T, D) -> (B, T, V_tgt)` logits.
# 

# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, track: bool = False):
        # x: (B, T, D), memory: (B, S, D)
        if track:
            print("  -- DecoderLayer --")
            show("dec layer in", x)
            show("memory (from enc)", memory)

        h = self.norm1(x)
        a = self.self_attn(h, h, h, attn_mask=tgt_mask, track=track, tag="dec-self")
        x = x + a
        if track:
            show("after dec self-attn+res", x)

        h = self.norm2(x)
        a = self.cross_attn(
            h, memory, memory, attn_mask=memory_mask, track=track, tag="dec-cross"
        )
        x = x + a
        if track:
            show("after cross-attn+res", x)

        h = self.norm3(x)
        x = x + self.ffn(h, track=track, tag="dec-ffn")
        if track:
            show("dec layer out", x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pe = SinusoidalPE(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None, track: bool = False):
        # tgt_ids: (B, T)
        if track:
            print("\n=== DECODER ===")
            show("tgt_ids", tgt_ids)

        x = self.embed(tgt_ids) * math.sqrt(D)  # (B, T, D)
        if track:
            show("tgt embed * sqrt(D)", x)
        x = self.pe(x)
        if track:
            show("tgt + PE", x)

        for i, layer in enumerate(self.layers):
            if track:
                print(f"  [decoder layer {i}]")
            x = layer(x, memory, tgt_mask, memory_mask, track=track)

        x = self.norm(x)                        # (B, T, D)
        if track:
            show("decoder hidden", x)

        logits = self.proj(x)                   # (B, T, V_tgt)
        if track:
            show("logits over vocab", logits)
        return logits


# %% [markdown]
# ## Masks + full seq2seq model
# 
# - **Pad mask**: `(B, 1, 1, L)` -- block attending to `PAD`
# - **Causal mask**: `(1, 1, T, T)` -- lower triangle for decoder self-attention
# - Combined target mask: pad **and** causal
# 

# %%
def make_pad_mask(ids: torch.Tensor) -> torch.Tensor:
    # ids: (B, L) -> (B, 1, 1, L)
    return (ids != PAD_ID).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    # (1, 1, L, L) lower-triangular
    m = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
    return m.unsqueeze(0).unsqueeze(0)


class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(V_SRC, D, H, D_FF, N_ENC)
        self.decoder = Decoder(V_TGT, D, H, D_FF, N_DEC)

    def forward(self, src_ids, tgt_ids, track: bool = False):
        src_mask = make_pad_mask(src_ids)                              # (B, 1, 1, S)
        tgt_pad = make_pad_mask(tgt_ids)                               # (B, 1, 1, T)
        causal = make_causal_mask(tgt_ids.size(1), tgt_ids.device)    # (1, 1, T, T)
        tgt_mask = tgt_pad & causal                                    # (B, 1, T, T)

        if track:
            print("\n=== MASKS ===")
            show("src_mask (pad)", src_mask)
            show("causal mask", causal)
            show("tgt_mask (pad & causal)", tgt_mask)

        memory = self.encoder(src_ids, src_mask, track=track)          # (B, S, D)
        logits = self.decoder(
            tgt_ids, memory, tgt_mask, src_mask, track=track
        )                                                              # (B, T, V)
        return logits


# %% [markdown]
# ## Run a forward pass and watch the shapes
# 

# %%
model = Seq2SeqTransformer()
model.eval()

src = torch.randint(1, V_SRC, (B, S))   # (B, S)
tgt = torch.randint(1, V_TGT, (B, T))   # (B, T)
src[1, -1] = PAD_ID                     # show padding in batch 1

with torch.no_grad():
    logits = model(src, tgt, track=True)

print("\n=== FINAL ===")
show("logits", logits)                 # (B, T, V_tgt)
show("greedy tokens", logits.argmax(-1))

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {n_params:,}")
print(
    """
Shape cheat-sheet:
  src_ids / tgt_ids          (B, S) / (B, T)
  embeddings + PE            (B, L, D)
  Q,K,V per head             (B, H, L, Dk)
  attention scores/weights   (B, H, Lq, Lk)
  encoder memory             (B, S, D)
  decoder logits             (B, T, V_tgt)
"""
)



