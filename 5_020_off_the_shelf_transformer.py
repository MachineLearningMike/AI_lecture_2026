"""
Machine translation by CALLING PyTorch's off-the-shelf building blocks --
not by writing attention / FFN from scratch (that was 151).

Pieces we import and stack:

  nn.TransformerEncoderLayer   -- one encoder block (self-attn + FFN)
  nn.TransformerEncoder        -- N copies of that layer
  nn.TransformerDecoderLayer   -- one decoder block (masked self-attn +
                                  cross-attn + FFN)
  nn.TransformerDecoder        -- N copies of that layer

Contrast with the lecture stack:
 -- MultiHeadAttention / FFN written by hand
 -- NMT on that hand-rolled stack
  153 (this file)              -- same EN->FR task, but assemble from
                                  torch.nn Encoder/Decoder Layer+stack APIs

What you still own:
  * token embeddings + positional encoding
  * BOS/EOS/PAD packing and masks
  * wiring encoder memory into the decoder
  * training loop (teacher forcing) and greedy decode
  * final Linear head to vocab logits

What the off-the-shelf layers own:
  * self-attention, cross-attention, FFN, residuals, LayerNorm

Shape notation (batch_first=True):
  B = batch, S = source length, T = target length, D = d_model, V = vocab

Run:
 conda run -n torch27 python
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Nested-tensor prototype notes on some PyTorch builds.
warnings.filterwarnings(
    "ignore",
    message=".*nested tensors is in prototype stage.*",
)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def show(name: str, x: torch.Tensor) -> torch.Tensor:
    print(f"  {name:32s} {tuple(x.shape)}")
    return x


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Bool upper-triangular mask: True = BLOCK future. Shape (L, L)."""
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
    )


# ---------------------------------------------------------------------------
# Same toy EN -> FR corpus as 152 (keep demos comparable)
# ---------------------------------------------------------------------------
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

DEMO_EN = "the cat eats fish"
DEMO_FR = "le chat mange du poisson"

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}


class Vocab:
    """Whitespace tokenizer. Ids: 0 PAD | 1 BOS | 2 EOS | 3 UNK | 4.. words."""

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
        out = []
        for i in ids:
            if skip_special and i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            out.append(self.itos.get(int(i), "<unk>"))
        return " ".join(out)


def pad_batch(seqs: list[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    """list[list[int]] -> LongTensor (B, L_max)."""
    bsz = len(seqs)
    length = max(len(s) for s in seqs)
    out = torch.full((bsz, length), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Positional encoding (PyTorch encoder/decoder layers do NOT include PE)
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                 # (L, D)
        pos = torch.arange(max_len).unsqueeze(1).float()   # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))        # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, L, D)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Off-the-shelf MT: assemble Encoder/Decoder from Layer + stack
# ---------------------------------------------------------------------------
class OffTheShelfMT(nn.Module):
    """
    Recipe (this is the whole point of 153):

      enc_layer = nn.TransformerEncoderLayer(...)
      encoder   = nn.TransformerEncoder(enc_layer, num_layers=N)

      dec_layer = nn.TransformerDecoderLayer(...)
      decoder   = nn.TransformerDecoder(dec_layer, num_layers=N)

    Then:
      memory = encoder(src_emb, src_key_padding_mask=...)
      out    = decoder(tgt_emb, memory, tgt_mask=..., ...)
      logits = Linear(out)

    We do NOT implement attention ourselves -- we CALL these modules.
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # --- embeddings we still write ourselves ---------------------------
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.pe = SinusoidalPE(d_model)

        # --- OFF-THE-SHELF ENCODER ----------------------------------------
        # Match 151: Pre-LN, no dropout (fairer comparison on the toy task).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, S, D)
            norm_first=True,
        )
        # Stack N identical layers (PyTorch clones the layer weights)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        # Prefer the plain dense path (skip nested-tensor prototype path).
        self.encoder.enable_nested_tensor = False

        # --- OFF-THE-SHELF DECODER ----------------------------------------
        # One layer template: masked self-attn + cross-attn + FFN
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, T, D)
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # --- vocab head we still write ourselves --------------------------
        self.out_proj = nn.Linear(d_model, tgt_vocab)

    def embed_src(self, src_ids: torch.Tensor, track: bool = False) -> torch.Tensor:
        """src_ids (B, S) -> (B, S, D) with PE."""
        if track:
            print("\n=== SRC EMBED (we own) ===")
            show("src_ids", src_ids)
        x = self.src_embed(src_ids) * math.sqrt(self.d_model)  # (B, S, D)
        if track:
            show("src embed * sqrt(D)", x)
        x = self.pe(x)
        if track:
            show("src + PE", x)
        return x

    def embed_tgt(self, tgt_ids: torch.Tensor, track: bool = False) -> torch.Tensor:
        """tgt_ids (B, T) -> (B, T, D) with PE."""
        if track:
            print("\n=== TGT EMBED (we own) ===")
            show("tgt_ids", tgt_ids)
        y = self.tgt_embed(tgt_ids) * math.sqrt(self.d_model)  # (B, T, D)
        if track:
            show("tgt embed * sqrt(D)", y)
        y = self.pe(y)
        if track:
            show("tgt + PE", y)
        return y

    def encode(
        self,
        src_ids: torch.Tensor,
        track: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          memory  (B, S, D)  -- encoder output
          src_pad (B, S)     -- True = PAD (for key_padding_mask)
        """
        src_emb = self.embed_src(src_ids, track=track)     # (B, S, D)
        src_pad = (src_ids == PAD_ID)                      # (B, S)

        if track:
            print("\n=== nn.TransformerEncoder (off-the-shelf) ===")
            show("src_emb", src_emb)
            show("src_key_padding_mask", src_pad)
            print("  True in padding mask = IGNORE that key position")

        # CALL the stack -- no hand-written attention here
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_pad,
        )                                                  # (B, S, D)
        if track:
            show("memory (encoder out)", memory)
        return memory, src_pad

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        track: bool = False,
    ) -> torch.Tensor:
        """
        Teacher-forced / partial decode.
        Returns hidden (B, T, D) before the vocab projection.
        """
        tgt_emb = self.embed_tgt(tgt_ids, track=track)     # (B, T, D)
        tgt_pad = (tgt_ids == PAD_ID)                      # (B, T)
        tgt_causal = causal_mask(tgt_ids.size(1), tgt_ids.device)  # (T, T)

        if track:
            print("\n=== nn.TransformerDecoder (off-the-shelf) ===")
            show("tgt_emb", tgt_emb)
            show("memory", memory)
            show("tgt_mask (causal)", tgt_causal)
            show("tgt_key_padding_mask", tgt_pad)
            show("memory_key_padding_mask", memory_key_padding_mask)
            print("  decoder layer = masked self-attn + cross-attn + FFN")

        # CALL the stack -- cross-attn to encoder memory is inside
        hidden = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_key_padding_mask,
        )                                                  # (B, T, D)
        if track:
            show("decoder hidden", hidden)
        return hidden

    def forward(
        self,
        src_ids: torch.Tensor,   # (B, S)
        tgt_ids: torch.Tensor,   # (B, T) teacher-forced decoder input
        track: bool = False,
    ) -> torch.Tensor:
        """logits (B, T, V_tgt)."""
        memory, src_pad = self.encode(src_ids, track=track)
        hidden = self.decode(tgt_ids, memory, src_pad, track=track)
        logits = self.out_proj(hidden)                     # (B, T, V)
        if track:
            print("\n=== VOCAB HEAD (we own) ===")
            show("logits", logits)
        return logits


# ---------------------------------------------------------------------------
# Train / decode
# ---------------------------------------------------------------------------
def teacher_forced_loss(
    model: OffTheShelfMT,
    src: torch.Tensor,   # (B, S)
    tgt: torch.Tensor,   # (B, T) BOS + tokens + EOS
    track: bool = False,
) -> torch.Tensor:
    """
    Decoder sees tgt[:, :-1]; predicts tgt[:, 1:].
      tgt_in (B, T-1) -> logits (B, T-1, V) vs labels (B, T-1)
    """
    if track:
        print("\n=== TEACHER FORCING ===")
        show("src", src)
        show("tgt full", tgt)

    tgt_in = tgt[:, :-1]     # (B, T-1)
    labels = tgt[:, 1:]      # (B, T-1)
    if track:
        show("tgt_in", tgt_in)
        show("labels", labels)

    logits = model(src, tgt_in, track=track)               # (B, T-1, V)
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=PAD_ID,
    )


@torch.no_grad()
def greedy_decode(
    model: OffTheShelfMT,
    src: torch.Tensor,          # (1, S)
    max_len: int = 20,
    track: bool = False,
) -> torch.Tensor:
    """Encode once with TransformerEncoder; step TransformerDecoder."""
    model.eval()
    if track:
        print("\n=== GREEDY DECODE ===")
        show("src", src)

    memory, src_pad = model.encode(src, track=track)       # (1, S, D), (1, S)
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=src.device)  # (1, 1)

    for step in range(max_len):
        hidden = model.decode(ys, memory, src_pad, track=False)  # (1, t, D)
        logits = model.out_proj(hidden)                    # (1, t, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        ys = torch.cat([ys, next_id], dim=1)               # (1, t+1)
        if track:
            print(f"  step {step}: id={int(next_id)} -> ys {tuple(ys.shape)}")
        if int(next_id) == EOS_ID:
            break

    if track:
        show("decoded ids", ys)
    return ys


def print_assembly_recipe() -> None:
    print(
        """
How 153 assembles an off-the-shelf Transformer
----------------------------------------------
  enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                         batch_first=True)
  encoder   = nn.TransformerEncoder(enc_layer, num_layers=N_enc)

  dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                         batch_first=True)
  decoder   = nn.TransformerDecoder(dec_layer, num_layers=N_dec)

  memory = encoder(src_emb,  src_key_padding_mask=src_pad)   # (B, S, D)
  hidden = decoder(tgt_emb, memory,
                   tgt_mask=causal,
                   tgt_key_padding_mask=tgt_pad,
                   memory_key_padding_mask=src_pad)          # (B, T, D)
  logits = Linear(hidden)                                    # (B, T, V)

vs 151: there we wrote MultiHeadAttention / FeedForward ourselves.
vs nn.Transformer(...): that is just a convenience wrapper around
         exactly these four classes. 153 calls them explicitly so the
         EncoderLayer / Encoder / DecoderLayer / Decoder roles are obvious.
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("153 Off-the-shelf Encoder/Decoder layers -- EN->FR")
    print("=" * 60)
    print_assembly_recipe()

    torch.manual_seed(0)

    src_vocab = Vocab([en for en, _ in PAIRS])
    tgt_vocab = Vocab([fr for _, fr in PAIRS])
    print(f"V_src={len(src_vocab)}  V_tgt={len(tgt_vocab)}")

    src_batch = pad_batch(
        [src_vocab.encode(en, add_bos=False, add_eos=True) for en, _ in PAIRS]
    )  # (N, S)
    tgt_batch = pad_batch(
        [tgt_vocab.encode(fr, add_bos=True, add_eos=True) for _, fr in PAIRS]
    )  # (N, T)
    show("src_batch", src_batch)
    show("tgt_batch", tgt_batch)

    model = OffTheShelfMT(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Trainable parameters: {n_params:,}")
    print(f"  nn.TransformerEncoder  : {n_enc:,}")
    print(f"  nn.TransformerDecoder  : {n_dec:,}")
    print(f"  embeds + PE + vocab head: {n_params - n_enc - n_dec:,}")

    # Show the actual module types once
    print("\nModule types:")
    print(f"  encoder      -> {type(model.encoder).__name__}")
    print(f"  encoder[0]   -> {type(model.encoder.layers[0]).__name__}")
    print(f"  decoder      -> {type(model.decoder).__name__}")
    print(f"  decoder[0]   -> {type(model.decoder.layers[0]).__name__}")

    # ---- train (overfit toy data, like 152) --------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    epochs = 400
    print(f"\nTraining {epochs} epochs on {len(PAIRS)} pairs...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = teacher_forced_loss(model, src_batch, tgt_batch, track=False)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"  epoch {ep:4d}  loss={loss.item():.4f}")

    # ---- tracked forward on the demo pair ---------------------------------
    demo_src = pad_batch([src_vocab.encode(DEMO_EN, add_bos=False, add_eos=True)])
    demo_tgt = pad_batch([tgt_vocab.encode(DEMO_FR, add_bos=True, add_eos=True)])
    model.eval()
    with torch.no_grad():
        loss = teacher_forced_loss(model, demo_src, demo_tgt, track=True)
    print(f"\nDemo teacher-forced loss: {loss.item():.4f}")

    # ---- greedy decode ----------------------------------------------------
    with torch.no_grad():
        hyp = greedy_decode(model, demo_src, track=True)
    print(f"\nEN:  {DEMO_EN}")
    print(f"REF: {DEMO_FR}")
    print(f"HYP: {tgt_vocab.decode(hyp[0])}")

    print(
        "\n"
        + "=" * 60
        + """
TAKEAWAY
  Build NMT by CALLING:
    TransformerEncoderLayer -> TransformerEncoder
    TransformerDecoderLayer -> TransformerDecoder
  Do not re-implement QKV attention unless you are teaching it (151).

ALIGNED WITH 151:
  Pre-LN (norm_first=True) and dropout=0.0 for a fairer toy-task comparison.

MASK GOTCHA (vs hand-rolled 151):
  151 pad mask:  1/True = KEEP
  153 pad mask:  True   = IGNORE   (src_key_padding_mask / tgt_key_padding_mask)
  153 causal:    bool upper triangle True = BLOCK future

nn.Transformer(...) is only a thin wrapper that builds these same four
classes for you. 153 makes the Layer vs stack split explicit.
"""
    )


if __name__ == "__main__":
    main()
