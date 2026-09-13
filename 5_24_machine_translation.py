"""
Tiny English -> French Neural Machine Translation using the encoder-decoder
a self-contained encoder-decoder Transformer.

What this lecture shows:
  1) Build source/target vocabs + special tokens (PAD/BOS/EOS/UNK)
  2) Encode a parallel corpus as id tensors  -- shapes tracked
  3) Teacher-forced training: predict tgt[t+1] from tgt[:t] + src memory
  4) Greedy decode one held-out / demo pair
  5) Visualize DECODER CROSS-ATTENTION  (target tokens x source tokens)

Cross-attention is the heart of classical NMT Transformers: each generated
French token looks at the English encoder memory to decide what to say next.

Run:
 conda run -n torch27 python
 MPLBACKEND=Agg conda run -n torch27 python
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F_torch

# ---------------------------------------------------------------------------
# Self-contained Transformer (inlined; no lecture-module import)
# ---------------------------------------------------------------------------
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
        # Populated every forward: (B, H, Lq, Lk) — used by 280_machine_translation
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
# Plot helpers (match lecture style)
# ---------------------------------------------------------------------------


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return (w / dpi * fraction, h / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


# Attention slides are registered here, then shown via run_slide_deck at end of main.
_ATTN_SLIDES: list = []


# ---------------------------------------------------------------------------
# Toy parallel corpus  (EN -> FR)
# ---------------------------------------------------------------------------
# Keep sentences short so the attention matrix is readable.
# The demo pair is intentionally somewhat word-aligned.
PAIRS = [
    ("the cat sat on the mat", "le chat s assit sur le tapis"),
    ("the dog sat on the mat", "le chien s assit sur le tapis"),
    ("the cat eats fish", "le chat mange du poisson"),
    ("the dog eats meat", "le chien mange de la viande"),
    ("I love cats", "j aime les chats"),
    ("I love dogs", "j aime les chiens"),
    ("she reads a book", "elle lit un livre"),
    ("he reads a book", "il lit un livre"),
    ("we drink water", "nous buvons de l eau"),
    ("they drink water", "ils boivent de l eau"),
]

# Focus pair for attention visualization (also in the training set -- overfit OK)
DEMO_EN = "the cat eats fish"
DEMO_FR = "le chat mange du poisson"

# Special tokens -- must stay consistent with PAD_ID == 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}


# ---------------------------------------------------------------------------
# Vocab + encode / decode
# ---------------------------------------------------------------------------
class Vocab:
    """Simple whitespace tokenizer + id map.

    Layout:
      0 PAD | 1 BOS | 2 EOS | 3 UNK | 4.. content
    """

    def __init__(self, sentences: list[str]):
        words = sorted({w for s in sentences for w in s.split()})
        self.itos = dict(SPECIAL)
        for i, w in enumerate(words, start=4):
            self.itos[i] = w
        self.stoi = {w: i for i, w in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, sentence: str, add_bos: bool, add_eos: bool) -> list[int]:
        ids = [self.stoi.get(w, UNK_ID) for w in sentence.split()]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks = []
        for i in ids:
            if skip_special and i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            toks.append(self.itos.get(int(i), "<unk>"))
        return " ".join(toks)


def pad_batch(seqs: list[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    """Pad variable-length id lists to (B, L_max)."""
    B = len(seqs)
    L = max(len(s) for s in seqs)
    out = torch.full((B, L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out  # (B, L)


# ---------------------------------------------------------------------------
# Training / decoding
# ---------------------------------------------------------------------------
def teacher_forced_loss(
    model: Seq2SeqTransformer,
    src: torch.Tensor,   # (B, S)  includes no BOS; plain source + optional EOS
    tgt: torch.Tensor,   # (B, T)  BOS + tokens + EOS
    track: bool = False,
) -> torch.Tensor:
    """Standard NMT teacher forcing.

    Input to decoder:  tgt[:, :-1]   = BOS + tokens        shape (B, T-1)
    Labels:            tgt[:,  1:]   = tokens + EOS        shape (B, T-1)
    Logits:            (B, T-1, V)
    """
    if track:
        print("\n=== TEACHER FORCING SHAPES ===")
        show("src (encoder input)", src)
        show("tgt (full with BOS/EOS)", tgt)

    tgt_in = tgt[:, :-1]     # (B, T-1)
    labels = tgt[:, 1:]      # (B, T-1)
    if track:
        show("tgt_in  (decoder in)", tgt_in)
        show("labels (next tokens)", labels)

    logits = model(src, tgt_in, track=track)   # (B, T-1, V)
    if track:
        show("logits", logits)

    # Flatten for cross-entropy: (B*(T-1), V) vs (B*(T-1),)
    V = logits.size(-1)
    loss = F_torch.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=PAD_ID,
    )
    return loss


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,          # (1, S)
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    max_len: int = 20,
    track: bool = False,
) -> torch.Tensor:
    """Autoregressive greedy decode. Returns token ids (1, T_out) including BOS."""
    model.eval()
    if track:
        print("\n=== GREEDY DECODE ===")
        show("src", src)

    # Encode once -- memory (1, S, D)
    src_mask = make_pad_mask(src)                          # (1, 1, 1, S)
    memory = model.encoder(src, src_mask, track=track)     # (1, S, D)
    if track:
        show("memory", memory)

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=src.device)  # (1, 1)
    for step in range(max_len):
        tgt_pad = make_pad_mask(ys)                        # (1, 1, 1, t)
        causal = make_causal_mask(ys.size(1), ys.device)   # (1, 1, t, t)
        tgt_mask = tgt_pad & causal
        logits = model.decoder(ys, memory, tgt_mask, src_mask, track=False)
        # logits: (1, t, V) -- take last time step
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        ys = torch.cat([ys, next_id], dim=1)               # (1, t+1)
        if track:
            print(f"  step {step}: append id={int(next_id)} -> ys {tuple(ys.shape)}")
        if int(next_id) == eos_id:
            break
    if track:
        show("decoded ids", ys)
    return ys


# ---------------------------------------------------------------------------
# Attention plots
# ---------------------------------------------------------------------------
def plot_cross_attention(
    weights: torch.Tensor,   # (H, T, S)  or (T, S)
    src_tokens: list[str],
    tgt_tokens: list[str],
    title: str,
):
    """Register heatmap slide(s) of decoder cross-attention for one pair."""
    weights = weights.detach().cpu().clone()
    src_tokens = list(src_tokens)
    tgt_tokens = list(tgt_tokens)

    if weights.ndim == 2:
        def build(weights=weights, src_tokens=src_tokens, tgt_tokens=tgt_tokens, title=title):
            fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
            fig.suptitle(title, fontsize=FS_FIGURE_TITLE - 4, fontweight="bold")
            im = ax.imshow(weights.numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(src_tokens)))
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right")
            ax.set_yticklabels(tgt_tokens)
            ax.set_xlabel("source (EN) keys", fontsize=FS_AXIS)
            ax.set_ylabel("target (FR) queries", fontsize=FS_AXIS)
            ax.set_title("cross-attention  (T x S)", fontsize=FS_AX_TITLE)
            for i in range(weights.size(0)):
                for j in range(weights.size(1)):
                    ax.text(
                        j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                        color="white" if weights[i, j] > 0.4 else "black",
                        fontsize=9,
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.text(
                0.5, 0.12,
                "Each row is a French query token; columns are English keys from encoder memory.\n"
                "Bright cell => that EN token strongly informs this FR token.",
                ha="center", fontsize=FS_ANNOT,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
            )
            fig.tight_layout(rect=[0, 0.14, 1, 0.90])
            return fig

        _ATTN_SLIDES.append(build)
        return

    def build_heads(weights=weights, src_tokens=src_tokens, tgt_tokens=tgt_tokens, title=title):
        H = weights.size(0)
        ncols = min(4, H)
        nrows = int(np.ceil(H / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=screen_figsize(), dpi=FIGURE_DPI)
        fig.suptitle(title, fontsize=FS_FIGURE_TITLE - 4, fontweight="bold")
        axes = np.array(axes).reshape(-1)
        im = None
        for h in range(H):
            ax = axes[h]
            im = ax.imshow(weights[h].numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(src_tokens)))
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=FS_ANNOT)
            ax.set_yticklabels(tgt_tokens, fontsize=FS_ANNOT)
            ax.set_title(f"head {h}", fontsize=FS_AX_TITLE - 4)
            if h % ncols == 0:
                ax.set_ylabel("FR query", fontsize=FS_ANNOT)
            if h >= H - ncols:
                ax.set_xlabel("EN key", fontsize=FS_ANNOT)
        for h in range(H, len(axes)):
            axes[h].axis("off")
        fig.colorbar(im, ax=axes[:H].tolist(), fraction=0.02, pad=0.02)
        fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.35, hspace=0.45)
        return fig

    _ATTN_SLIDES.append(build_heads)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Machine Translation -- EN->FR with Transformer")
    print("=" * 60)

    torch.manual_seed(0)

    # --- vocabs ----------------------------------------------------------
    src_vocab = Vocab([en for en, _ in PAIRS])
    tgt_vocab = Vocab([fr for _, fr in PAIRS])
    print(f"Source vocab size V_src = {len(src_vocab)}  (PAD/BOS/EOS/UNK + words)")
    print(f"Target vocab size V_tgt = {len(tgt_vocab)}")

    # --- encode corpus ---------------------------------------------------
    # Source: tokens + EOS   (no BOS needed for encoder)
    # Target: BOS + tokens + EOS
    src_seqs = [src_vocab.encode(en, add_bos=False, add_eos=True) for en, _ in PAIRS]
    tgt_seqs = [tgt_vocab.encode(fr, add_bos=True, add_eos=True) for _, fr in PAIRS]
    src_batch = pad_batch(src_seqs)   # (N, S)
    tgt_batch = pad_batch(tgt_seqs)   # (N, T)
    print("\nCorpus tensors:")
    show("src_batch", src_batch)
    show("tgt_batch", tgt_batch)
    print(f"  example src ids[0]: {src_batch[0].tolist()} -> '{src_vocab.decode(src_batch[0])}'")
    print(f"  example tgt ids[0]: {tgt_batch[0].tolist()} -> '{tgt_vocab.decode(tgt_batch[0])}'")

    # --- model -----------------------------------------------------------
    # Small but enough to overfit the toy corpus and show alignments
    d_model, n_heads, d_ff, n_layers = 64, 4, 128, 2
    model = Seq2SeqTransformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_enc=n_layers,
        n_dec=n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nModel: d_model={d_model}, heads={n_heads}, d_ff={d_ff}, "
        f"layers={n_layers}/{n_layers}, params={n_params:,}"
    )

    # --- train (overfit toy data; lecture demo, not a real MT system) ----
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    epochs = 400
    print(f"\nTraining {epochs} epochs on {len(PAIRS)} pairs (expect near-memorization)...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = teacher_forced_loss(model, src_batch, tgt_batch, track=False)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"  epoch {ep:4d}  loss={loss.item():.4f}")

    # --- one tracked teacher-forced forward on the DEMO pair -------------
    demo_src = pad_batch([src_vocab.encode(DEMO_EN, add_bos=False, add_eos=True)])
    demo_tgt = pad_batch([tgt_vocab.encode(DEMO_FR, add_bos=True, add_eos=True)])
    model.eval()
    with torch.no_grad():
        _ = teacher_forced_loss(model, demo_src, demo_tgt, track=True)

    # Cross-attn from last decoder layer after the tracked forward:
    #   last_attn_weights: (B=1, H, T_in, S) where T_in = len(tgt_in) = T-1
    cross = model.cross_attn_weights(layer=-1)
    assert cross is not None
    show("cross-attn last layer", cross)          # (1, H, T-1, S)

    # Mean over heads for a single readable alignment matrix: (T-1, S)
    cross_mean = cross[0].mean(dim=0)             # (T-1, S)
    show("cross-attn mean heads", cross_mean)

    # Token labels for the heatmap
    # Decoder queries correspond to tgt_in = BOS + FR tokens  (no final EOS as query
    # when predicting EOS -- but we include whatever length T-1 has)
    tgt_in_ids = demo_tgt[0, :-1].tolist()
    src_ids = demo_src[0].tolist()
    # Drop trailing PAD on source for plotting
    s_len = int((demo_src[0] != PAD_ID).sum())
    t_len = int((demo_tgt[0, :-1] != PAD_ID).sum())
    src_toks = [src_vocab.itos[i] for i in src_ids[:s_len]]
    tgt_toks = [tgt_vocab.itos[i] for i in tgt_in_ids[:t_len]]
    # Trim attention to non-pad region: (t_len, s_len)
    attn_mean = cross_mean[:t_len, :s_len].cpu()
    attn_heads = cross[0, :, :t_len, :s_len].cpu()  # (H, t_len, s_len)

    print("\nDemo pair:")
    print(f"  EN: {DEMO_EN}")
    print(f"  FR: {DEMO_FR}")
    print(f"  src tokens: {src_toks}")
    print(f"  tgt query tokens (decoder in): {tgt_toks}")

    # --- greedy decode ---------------------------------------------------
    with torch.no_grad():
        hyp = greedy_decode(model, demo_src, track=True)
    print(f"\nGreedy translation: '{tgt_vocab.decode(hyp[0])}'")
    print(f"  (reference FR:     '{DEMO_FR}')")

    # --- plots -----------------------------------------------------------
    plot_cross_attention(
        attn_mean,
        src_toks,
        tgt_toks,
        title=(
            "Cross-attention (mean over heads)\n"
            f"'{DEMO_EN}'  ->  '{DEMO_FR}'"
        ),
    )
    plot_cross_attention(
        attn_heads,
        src_toks,
        tgt_toks,
        title=(
            "Cross-attention per head\n"
            f"'{DEMO_EN}'  ->  '{DEMO_FR}'"
        ),
    )

    print(
        "\nTakeaway:\n"
        "  Encoder writes a memory of the English sentence (B, S, D).\n"
        "  Decoder generates French left-to-right; at each step CROSS-ATTENTION\n"
        "  reads from that memory. The heatmap is literally those softmax weights.\n"
        "  Bright diagonals / peaks near translations of the same word are typical\n"
        "  after the model memorizes a tiny aligned corpus."
    )

    if _ATTN_SLIDES:
        run_slide_deck(list(_ATTN_SLIDES))
        _ATTN_SLIDES.clear()


if __name__ == "__main__":
    main()
