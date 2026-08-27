"""
152_machine_translation.py

Tiny English -> French Neural Machine Translation using the encoder-decoder
Transformer classes from 151_transformer_code.py.

What this lecture shows:
  1) Build source/target vocabs + special tokens (PAD/BOS/EOS/UNK)
  2) Encode a parallel corpus as id tensors  -- shapes tracked
  3) Teacher-forced training: predict tgt[t+1] from tgt[:t] + src memory
  4) Greedy decode one held-out / demo pair
  5) Visualize DECODER CROSS-ATTENTION  (target tokens x source tokens)

Cross-attention is the heart of classical NMT Transformers: each generated
French token looks at the English encoder memory to decide what to say next.

Companion:
  150_transformer.py      -- attention intuition
  151_transformer_code.py -- encoder / decoder implementation

Run:
  conda run -n torch27 python 152_machine_translation.py
  MPLBACKEND=Agg conda run -n torch27 python 152_machine_translation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_torch

# Make sure we import the sibling 151 module from this folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from importlib import import_module

t151 = import_module("151_transformer_code")

Seq2SeqTransformer = t151.Seq2SeqTransformer
make_pad_mask = t151.make_pad_mask
make_causal_mask = t151.make_causal_mask
show = t151.show
PAD_ID = t151.PAD_ID  # 0


# ---------------------------------------------------------------------------
# Plot helpers (match lecture style)
# ---------------------------------------------------------------------------
FS_FIGURE_TITLE = 28
FS_AX_TITLE = 22
FS_AXIS = 18
FS_ANNOT = 12
FIGURE_DPI = 150
SCREEN_FRACTION = 0.95

plt.rcParams.update({
    "figure.titlesize": FS_FIGURE_TITLE,
    "axes.titlesize": FS_AX_TITLE,
    "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_AXIS,
    "ytick.labelsize": FS_AXIS,
    "figure.dpi": FIGURE_DPI,
})


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


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.frame.Maximize(True)
                except Exception:
                    pass
    plt.show()


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

# Special tokens -- must stay consistent with 151 PAD_ID == 0
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
    """Heatmap(s) of decoder cross-attention for one translation pair."""
    if weights.ndim == 2:
        # Single matrix (T, S)
        fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
        fig.suptitle(title, fontsize=FS_FIGURE_TITLE, fontweight="bold")
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
            0.5, 0.02,
            "Each row is a French query token; columns are English keys from encoder memory.\n"
            "Bright cell => that EN token strongly informs this FR token.",
            ha="center", fontsize=FS_ANNOT,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        show_maximized(fig)
        return

    # Per-head panels: weights (H, T, S)
    H = weights.size(0)
    ncols = min(4, H)
    nrows = int(np.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(title, fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    axes = np.array(axes).reshape(-1)
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
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.35, hspace=0.45)
    show_maximized(fig)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("152 Machine Translation -- EN->FR with 151 Transformer")
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
        title=f"Cross-attention (mean heads):  '{DEMO_EN}'  ->  '{DEMO_FR}'",
    )
    plot_cross_attention(
        attn_heads,
        src_toks,
        tgt_toks,
        title=f"Cross-attention per head:  '{DEMO_EN}'  ->  '{DEMO_FR}'",
    )

    print(
        "\nTakeaway:\n"
        "  Encoder writes a memory of the English sentence (B, S, D).\n"
        "  Decoder generates French left-to-right; at each step CROSS-ATTENTION\n"
        "  reads from that memory. The heatmap is literally those softmax weights.\n"
        "  Bright diagonals / peaks near translations of the same word are typical\n"
        "  after the model memorizes a tiny aligned corpus."
    )


if __name__ == "__main__":
    main()
