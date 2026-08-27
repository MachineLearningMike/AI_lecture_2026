"""
200_transformer_code.py

Minimal PyTorch Transformer encoder + decoder with SHAPE TRACKING.

Companion intuition: 190_transformer.py
Positional encoding:   260_positional_encoding.py

Notation used in comments:
  B  = batch size
  S  = source (encoder) sequence length
  T  = target (decoder) sequence length
  D  = d_model  (model / embedding width)
  H  = number of attention heads
  Dk = D // H   (per-head key/query/value width)
  V  = vocab size
  D_FF = d_ff   (feed-forward hidden width)

Run:
  conda run -n torch27 python 200_transformer_code.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
B = 2          # batch
S = 5          # source length  (e.g. "the cat sat on mat")
T = 4          # target length  (e.g. translation prefix)
D = 32         # d_model
H = 4          # n_heads
Dk = D // H    # per-head dim  (must divide evenly)
D_FF = 64      # d_ff  (named D_FF so it does not shadow torch.nn.functional as F)
V_SRC = 50     # source vocab
V_TGT = 60     # target vocab
N_ENC = 2      # encoder layers
N_DEC = 2      # decoder layers
PAD_ID = 0

assert D % H == 0, "d_model must be divisible by n_heads"
torch.manual_seed(0)


def show(name: str, x: torch.Tensor) -> torch.Tensor:
    """Print tensor shape (and optionally peek) then return x unchanged."""
    print(f"  {name:28s} {tuple(x.shape)}")
    return x


# ---------------------------------------------------------------------------
# Positional encoding  (sinusoidal, added to embeddings)
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # (L, D)
        pos = torch.arange(max_len).unsqueeze(1)    # (L, 1)
        # div_term_i = 10000^{2i/D}
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Multi-head attention (self or cross)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.H = n_heads
        self.Dk = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # Populated every forward: (B, H, Lq, Lk) — used by 210_machine_translation
        self.last_attn_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dk)
        B, L, _ = x.shape
        x = x.view(B, L, self.H, self.Dk).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dk) -> (B, L, D)
        B, H, L, Dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dk)

    def forward(
        self,
        query: torch.Tensor,   # (B, Lq, D)
        key: torch.Tensor,     # (B, Lk, D)
        value: torch.Tensor,   # (B, Lk, D)
        attn_mask: torch.Tensor | None = None,  # broadcastable to (B, H, Lq, Lk)
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

        # scores: (B, H, Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.Dk)
        if track:
            show(f"{tag} scores QK^T/sqrt", scores)

        if attn_mask is not None:
            # True (or 1) = keep; we use additive mask with -inf on blocked
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)      # (B, H, Lq, Lk)
        self.last_attn_weights = weights.detach()
        if track:
            show(f"{tag} softmax weights", weights)

        ctx = torch.matmul(weights, V)           # (B, H, Lq, Dk)
        if track:
            show(f"{tag} weighted V", ctx)

        out = self.W_o(self._merge_heads(ctx))   # (B, Lq, D)
        if track:
            show(f"{tag} out (merged)", out)
        return out


# ---------------------------------------------------------------------------
# Position-wise FFN
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, track: bool = False, tag: str = "ffn") -> torch.Tensor:
        # x: (B, L, D) -> hidden (B, L, D_FF) -> (B, L, D)
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


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                 # (B, S, D)
        src_mask: torch.Tensor | None,   # (B, 1, 1, S) or (B, 1, S, S)
        track: bool = False,
    ) -> torch.Tensor:
        if track:
            print("  -- EncoderLayer --")
            show("enc layer in", x)

        # Pre-norm style residual: x + Attn(LN(x))
        h = self.norm1(x)
        # Self-attention: Q=K=V=h  -> (B, S, D)
        a = self.self_attn(h, h, h, attn_mask=src_mask, track=track, tag="enc-self")
        x = x + a
        if track:
            show("after enc self-attn+res", x)

        h = self.norm2(x)
        x = x + self.ffn(h, track=track, tag="enc-ffn")
        if track:
            show("enc layer out", x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pe = SinusoidalPE(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src_ids: torch.Tensor,           # (B, S)
        src_mask: torch.Tensor | None,
        track: bool = False,
    ) -> torch.Tensor:
        if track:
            print("\n=== ENCODER ===")
            show("src_ids", src_ids)

        x = self.embed(src_ids) * math.sqrt(self.d_model)   # (B, S, D)
        if track:
            show("src embed * sqrt(D)", x)
        x = self.pe(x)                           # (B, S, D)
        if track:
            show("src + PE", x)

        for i, layer in enumerate(self.layers):
            if track:
                print(f"  [encoder layer {i}]")
            x = layer(x, src_mask, track=track)

        x = self.norm(x)                         # (B, S, D)
        if track:
            show("encoder memory", x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, D)  decoder states
        memory: torch.Tensor,              # (B, S, D)  encoder output
        tgt_mask: torch.Tensor | None,     # causal (+ pad)  (B, 1, T, T)
        memory_mask: torch.Tensor | None,  # src pad         (B, 1, 1, S)
        track: bool = False,
    ) -> torch.Tensor:
        if track:
            print("  -- DecoderLayer --")
            show("dec layer in", x)
            show("memory (from enc)", memory)

        # 1) Masked self-attention over target (no future)
        h = self.norm1(x)
        a = self.self_attn(h, h, h, attn_mask=tgt_mask, track=track, tag="dec-self")
        x = x + a
        if track:
            show("after dec self-attn+res", x)

        # 2) Cross-attention: query=decoder, key/value=encoder memory
        h = self.norm2(x)
        a = self.cross_attn(
            h, memory, memory, attn_mask=memory_mask, track=track, tag="dec-cross",
        )
        x = x + a
        if track:
            show("after cross-attn+res", x)

        # 3) FFN
        h = self.norm3(x)
        x = x + self.ffn(h, track=track, tag="dec-ffn")
        if track:
            show("dec layer out", x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pe = SinusoidalPE(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab)  # to vocab logits

    def forward(
        self,
        tgt_ids: torch.Tensor,             # (B, T)
        memory: torch.Tensor,              # (B, S, D)
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        track: bool = False,
    ) -> torch.Tensor:
        if track:
            print("\n=== DECODER ===")
            show("tgt_ids", tgt_ids)

        x = self.embed(tgt_ids) * math.sqrt(self.d_model)   # (B, T, D)
        if track:
            show("tgt embed * sqrt(D)", x)
        x = self.pe(x)                           # (B, T, D)
        if track:
            show("tgt + PE", x)

        for i, layer in enumerate(self.layers):
            if track:
                print(f"  [decoder layer {i}]")
            x = layer(x, memory, tgt_mask, memory_mask, track=track)

        x = self.norm(x)                         # (B, T, D)
        if track:
            show("decoder hidden", x)

        logits = self.proj(x)                    # (B, T, V_tgt)
        if track:
            show("logits over vocab", logits)
        return logits


# ---------------------------------------------------------------------------
# Full encoder-decoder model
# ---------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int = V_SRC,
        tgt_vocab: int = V_TGT,
        d_model: int = D,
        n_heads: int = H,
        d_ff: int = D_FF,
        n_enc: int = N_ENC,
        n_dec: int = N_DEC,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_enc)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_dec)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        track: bool = False,
    ) -> torch.Tensor:
        src_mask = make_pad_mask(src_ids)                 # (B, 1, 1, S)
        tgt_pad = make_pad_mask(tgt_ids)                  # (B, 1, 1, T)
        causal = make_causal_mask(tgt_ids.size(1), tgt_ids.device)  # (1, 1, T, T)
        tgt_mask = tgt_pad & causal                       # (B, 1, T, T) broadcast

        if track:
            print("\n=== MASKS ===")
            show("src_mask (pad)", src_mask)
            show("causal mask", causal)
            show("tgt_mask (pad & causal)", tgt_mask)

        memory = self.encoder(src_ids, src_mask, track=track)
        logits = self.decoder(tgt_ids, memory, tgt_mask, src_mask, track=track)
        return logits

    def cross_attn_weights(self, layer: int = -1) -> torch.Tensor | None:
        """Return last decoder cross-attention weights (B, H, T, S), if any."""
        return self.decoder.layers[layer].cross_attn.last_attn_weights

    def self_attn_weights_dec(self, layer: int = -1) -> torch.Tensor | None:
        """Return last decoder masked self-attention weights (B, H, T, T)."""
        return self.decoder.layers[layer].self_attn.last_attn_weights

    def self_attn_weights_enc(self, layer: int = -1) -> torch.Tensor | None:
        """Return last encoder self-attention weights (B, H, S, S)."""
        return self.encoder.layers[layer].self_attn.last_attn_weights



# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def make_pad_mask(ids: torch.Tensor) -> torch.Tensor:
    """1 where token is real, 0 where PAD. Shape -> (B, 1, 1, L) for broadcasting."""
    # ids: (B, L)
    mask = (ids != PAD_ID).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
    return mask


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular ones: position i may attend to j <= i. (1, 1, L, L)."""
    m = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
    return m.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("151 Transformer code -- encoder + decoder shape tour")
    print("=" * 60)
    print(
        f"Config: B={B}, S={S}, T={T}, D={D}, H={H}, Dk={Dk}, D_FF={D_FF}, "
        f"V_src={V_SRC}, V_tgt={V_TGT}, N_enc={N_ENC}, N_dec={N_DEC}"
    )

    model = Seq2SeqTransformer()
    model.eval()

    # Fake token ids (avoid PAD=0 for clarity in the demo)
    src = torch.randint(1, V_SRC, (B, S))
    tgt = torch.randint(1, V_TGT, (B, T))
    # Put a pad in batch 1 source to show mask effect
    src[1, -1] = PAD_ID

    with torch.no_grad():
        logits = model(src, tgt, track=True)

    print("\n=== FINAL ===")
    show("logits", logits)
    pred = logits.argmax(dim=-1)
    show("greedy tokens", pred)

    # Parameter count (sanity)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {n_params:,}")
    print(
        "\nShape cheat-sheet:\n"
        "  src_ids / tgt_ids          (B, S) / (B, T)\n"
        "  embeddings + PE            (B, L, D)\n"
        "  Q,K,V per head             (B, H, L, Dk)\n"
        "  attention scores/weights   (B, H, Lq, Lk)\n"
        "  encoder memory             (B, S, D)\n"
        "  decoder logits             (B, T, V_tgt)"
    )


if __name__ == "__main__":
    main()
