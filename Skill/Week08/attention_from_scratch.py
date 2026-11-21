"""
Module II Lab: Scaled Dot-Product Attention and Multi-Head Attention from Scratch

Implements both from first principles in PyTorch, then proves correctness by
checking every tensor against torch's own reference implementations
(F.scaled_dot_product_attention and nn.MultiheadAttention).

Run:  python attention_from_scratch.py
      python attention_from_scratch.py --seed 7 --no-csv
"""

import argparse
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CSV_FILENAME = "attention_results.csv"
RULE = "=" * 84
THIN = "-" * 84


# ----------------------------------------------------------------------------
# 1. Scaled dot-product attention
# ----------------------------------------------------------------------------

def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0,
                                 return_weights=True):
    """Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    query : (..., L_q, d_k)   what each position is looking for
    key   : (..., L_k, d_k)   what each position offers as a label
    value : (..., L_k, d_v)   what each position actually contributes
    mask  : broadcastable bool tensor, True = keep, False = forbid

    Returns (output, weights) with output (..., L_q, d_v)
    and weights (..., L_q, L_k) summing to 1 along the last axis.
    """
    d_k = query.size(-1)

    # Step 1 - raw similarity between every query and every key.
    scores = query @ key.transpose(-2, -1)            # (..., L_q, L_k)

    # Step 2 - scale. Without this the variance of the dot product grows with
    # d_k, softmax saturates, and gradients vanish. See report_scaling_effect.
    scores = scores / math.sqrt(d_k)

    # Step 3 - mask before softmax, never after. -inf becomes exactly 0 after
    # exponentiation, so forbidden positions cannot leak probability mass.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 4 - softmax turns scores into a distribution over the key positions.
    weights = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    # Step 5 - weighted sum of the values.
    output = weights @ value                          # (..., L_q, d_v)

    return (output, weights) if return_weights else output


# ----------------------------------------------------------------------------
# 2. Multi-head attention
# ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """h attention heads run in parallel on d_model / h dimensions each.

    One big head of size d_model can only average one way. h smaller heads can
    each attend to a different relationship (syntax, coreference, position) and
    the output projection recombines them.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout

        # One projection per role. Each is d_model -> d_model and is split into
        # heads afterwards, which is cheaper than h separate small matrices.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.last_weights = None   # kept for inspection after a forward pass

    def _split_heads(self, x):
        """(B, L, d_model) -> (B, h, L, d_head)"""
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(B, h, L, d_head) -> (B, L, d_model)"""
        B, h, L, d_head = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, h * d_head)

    def forward(self, query, key, value, mask=None, need_weights=True):
        """mask broadcasts to (B, h, L_q, L_k); True = keep, False = forbid."""
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if mask is not None and mask.dim() == 2:
            # (L_q, L_k) -> (1, 1, L_q, L_k) so it applies to every batch and head
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask is not None and mask.dim() == 3:
            # (B, L_q, L_k) -> (B, 1, L_q, L_k)
            mask = mask.unsqueeze(1)

        dropout_p = self.dropout if self.training else 0.0
        attn_out, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=dropout_p
        )

        out = self.w_o(self._merge_heads(attn_out))
        self.last_weights = weights.detach()

        return (out, weights) if need_weights else out


# ----------------------------------------------------------------------------
# 3. Masks
# ----------------------------------------------------------------------------

def causal_mask(seq_len: int, device=None):
    """Lower-triangular keep-mask: position i may attend to 0..i only.

    This single tensor is the whole difference between an encoder (BERT, sees
    everything) and a decoder (GPT, sees only the past).
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def padding_mask(lengths, max_len: int, device=None):
    """(B,) real lengths -> (B, 1, max_len) keep-mask over key positions."""
    lengths = torch.as_tensor(lengths, device=device)
    ar = torch.arange(max_len, device=device)
    return (ar[None, :] < lengths[:, None]).unsqueeze(1)


# ----------------------------------------------------------------------------
# 4. Reporting helpers
# ----------------------------------------------------------------------------

def heading(title: str):
    print("\n" + RULE)
    print(title)
    print(RULE)


def show_matrix(mat, row_labels=None, col_labels=None, width=7, dp=3):
    """Print a small 2-D tensor as a readable table."""
    mat = mat.detach().cpu()
    n_rows, n_cols = mat.shape
    row_labels = row_labels or [f"r{i}" for i in range(n_rows)]
    col_labels = col_labels or [f"c{j}" for j in range(n_cols)]

    header = " " * 12 + "".join(f"{c:>{width}}" for c in col_labels)
    print(header)
    for i in range(n_rows):
        cells = "".join(f"{mat[i, j].item():>{width}.{dp}f}" for j in range(n_cols))
        print(f"{row_labels[i]:<12}{cells}")


def check(label: str, ours, reference, tol=1e-5, rows=None):
    """Compare our tensor against torch's and print the max absolute difference."""
    diff = (ours - reference).abs().max().item()
    ok = diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:<52} max|diff| = {diff:.3e}")
    if rows is not None:
        rows.append([label, f"{diff:.3e}", status])
    return ok


# ----------------------------------------------------------------------------
# 5. Demonstrations
# ----------------------------------------------------------------------------

def demo_single_head(rows):
    heading("PART 1  SCALED DOT-PRODUCT ATTENTION ON A TOY SENTENCE")

    tokens = ["The", "cat", "sat", "on", "it"]
    L, d_k = len(tokens), 4

    torch.manual_seed(0)
    q = torch.randn(L, d_k)
    k = torch.randn(L, d_k)
    v = torch.randn(L, d_k)

    print(f"\nTokens        : {tokens}")
    print(f"Q, K, V shape : {tuple(q.shape)}  (L={L}, d_k={d_k})")

    scores_raw = q @ k.transpose(-2, -1)
    scores_scaled = scores_raw / math.sqrt(d_k)

    print(f"\n[STEP 1] RAW SCORES  Q @ K^T   (before scaling)")
    print(THIN)
    show_matrix(scores_raw, tokens, tokens)

    print(f"\n[STEP 2] SCALED SCORES  / sqrt(d_k) = / {math.sqrt(d_k):.4f}")
    print(THIN)
    show_matrix(scores_scaled, tokens, tokens)

    out, weights = scaled_dot_product_attention(q, k, v)

    print(f"\n[STEP 3] ATTENTION WEIGHTS  softmax(scaled scores)")
    print(THIN)
    show_matrix(weights, tokens, tokens)
    print(f"\n  Each row sums to 1: {weights.sum(-1).tolist()}")
    print(f"  Output shape      : {tuple(out.shape)}  (same as V)")

    print(f"\n[STEP 4] WHERE EACH TOKEN LOOKS (argmax of its row)")
    print(THIN)
    for i, tok in enumerate(tokens):
        j = weights[i].argmax().item()
        print(f"  {tok:<6} attends most to {tokens[j]:<6} ({weights[i, j].item()*100:5.1f}%)")

    ref = F.scaled_dot_product_attention(q, k, v)
    check("single-head output vs F.scaled_dot_product_attention", out, ref, rows=rows)


def demo_scaling_effect(rows):
    heading("PART 2  WHY DIVIDE BY sqrt(d_k)")

    print("\nQ and K are random with unit variance, so each dot product is a sum")
    print("of d_k independent products and its variance grows as d_k. Large")
    print("scores saturate softmax: one weight goes to 1, the rest to 0, and the")
    print("gradient through softmax vanishes.\n")

    print(f"{'d_k':>6} | {'std(raw)':>10} | {'std(scaled)':>12} | {'max w unscaled':>15} | {'max w scaled':>13}")
    print(THIN)

    torch.manual_seed(0)
    for d_k in (4, 16, 64, 256, 1024):
        q = torch.randn(1, 64, d_k)
        k = torch.randn(1, 64, d_k)

        raw = q @ k.transpose(-2, -1)
        scaled = raw / math.sqrt(d_k)

        w_unscaled = torch.softmax(raw, dim=-1).max().item()
        w_scaled = torch.softmax(scaled, dim=-1).max().item()

        print(f"{d_k:>6} | {raw.std().item():>10.3f} | {scaled.std().item():>12.3f} "
              f"| {w_unscaled:>15.4f} | {w_scaled:>13.4f}")

        rows.append([f"scaling d_k={d_k}", f"std_raw={raw.std().item():.3f}", "INFO"])

    print("\n  std(raw) tracks sqrt(d_k); std(scaled) stays near 1 at every size.")
    print("  Unscaled, the largest weight approaches 1.0 as d_k grows - a hard")
    print("  argmax with no usable gradient. Scaled, it stays moderate.")


def demo_causal_mask(rows):
    heading("PART 3  CAUSAL MASK - THE DIFFERENCE BETWEEN BERT AND GPT")

    tokens = ["The", "cat", "sat", "on", "it"]
    L, d_k = len(tokens), 8

    torch.manual_seed(1)
    q, k, v = torch.randn(3, L, d_k)

    mask = causal_mask(L)
    print("\nKeep-mask (1 = may attend, 0 = forbidden):")
    print(THIN)
    show_matrix(mask.float(), tokens, tokens, width=7, dp=0)

    _, w_open = scaled_dot_product_attention(q, k, v)
    _, w_causal = scaled_dot_product_attention(q, k, v, mask=mask)

    print("\nUnmasked weights (encoder / BERT - every token sees the future):")
    print(THIN)
    show_matrix(w_open, tokens, tokens)

    print("\nCausally masked weights (decoder / GPT - strictly upper triangle is 0):")
    print(THIN)
    show_matrix(w_causal, tokens, tokens)

    upper = w_causal.triu(diagonal=1)
    print(f"\n  Total weight leaked into the future: {upper.sum().item():.3e}")
    print(f"  Row sums still 1 after masking     : {w_causal.sum(-1).tolist()}")
    print("  Token 0 has only itself to look at, so its weight is exactly 1.0.")

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ours, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    check("causal output vs F.sdpa(is_causal=True)", out_ours, ref, rows=rows)
    rows.append(["future leakage", f"{upper.sum().item():.3e}",
                 "PASS" if upper.sum().item() < 1e-9 else "FAIL"])


def demo_padding_mask(rows):
    heading("PART 4  PADDING MASK - IGNORING FILLER IN A BATCH")

    B, L, d_k = 2, 5, 8
    lengths = [5, 3]

    torch.manual_seed(2)
    q, k, v = torch.randn(3, B, L, d_k)

    mask = padding_mask(lengths, L)
    print(f"\nBatch of {B} sequences padded to length {L}; real lengths {lengths}")
    print("\nKeep-mask over key positions:")
    print(THIN)
    show_matrix(mask.squeeze(1).float(), [f"seq{i}" for i in range(B)],
                [f"p{j}" for j in range(L)], width=7, dp=0)

    _, w = scaled_dot_product_attention(q, k, v, mask=mask)

    print("\nSequence 1 weights (positions 3 and 4 are padding):")
    print(THIN)
    show_matrix(w[1], [f"q{i}" for i in range(L)], [f"k{j}" for j in range(L)])

    pad_mass = w[1][:, 3:].sum().item()
    print(f"\n  Weight given to padded keys: {pad_mass:.3e}")
    print("  Without this mask the model averages in meaningless vectors and")
    print("  the same sentence gets a different encoding at a different batch size.")
    rows.append(["padding leakage", f"{pad_mass:.3e}",
                 "PASS" if pad_mass < 1e-9 else "FAIL"])


def demo_multi_head(rows):
    heading("PART 5  MULTI-HEAD ATTENTION")

    B, L, d_model, h = 2, 6, 64, 8
    torch.manual_seed(3)

    mha = MultiHeadAttention(d_model, h)
    mha.eval()
    x = torch.randn(B, L, d_model)

    print(f"\n  d_model   = {d_model}")
    print(f"  num_heads = {h}")
    print(f"  d_head    = {d_model // h}   (d_model / num_heads)")
    print(f"  input     = {tuple(x.shape)}")

    out, w = mha(x, x, x)

    print(f"\n  weights   = {tuple(w.shape)}   (B, heads, L_q, L_k)")
    print(f"  output    = {tuple(out.shape)}   (same shape as input)")

    n_params = sum(p.numel() for p in mha.parameters())
    print(f"\n  Parameters: {n_params:,}  =  4 projections x ({d_model}x{d_model} + {d_model})")
    print("  Head count does NOT change the parameter count - it only changes")
    print("  how the same d_model dimensions are partitioned.")

    print("\n[HEADS DISAGREE]  Row 0 of the weight matrix, per head:")
    print(THIN)
    print(" " * 10 + "".join(f"{f'k{j}':>8}" for j in range(L)))
    for head in range(h):
        cells = "".join(f"{w[0, head, 0, j].item():>8.3f}" for j in range(L))
        print(f"head {head:<5}{cells}")

    spread = w[0, :, 0, :].std(dim=0).mean().item()
    print(f"\n  Mean std across heads for this row: {spread:.4f}")
    print("  Non-zero spread is the point: each head learned a different view.")

    # --- Verify against nn.MultiheadAttention by copying our weights across ---
    print("\n[VERIFICATION AGAINST torch.nn.MultiheadAttention]")
    print(THIN)

    ref = nn.MultiheadAttention(d_model, h, batch_first=True, bias=True)
    ref.eval()
    with torch.no_grad():
        # torch packs q, k, v projections into one (3*d_model, d_model) matrix
        ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0))
        ref.in_proj_bias.copy_(torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias], dim=0))
        ref.out_proj.weight.copy_(mha.w_o.weight)
        ref.out_proj.bias.copy_(mha.w_o.bias)

    with torch.no_grad():
        ours, ours_w = mha(x, x, x)
        ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=True)

    check("multi-head output vs nn.MultiheadAttention", ours, ref_out, rows=rows)
    check("attention weights (averaged over heads)", ours_w.mean(dim=1), ref_w, rows=rows)

    # Causal multi-head
    cm = causal_mask(L)
    with torch.no_grad():
        ours_c, _ = mha(x, x, x, mask=cm)
        ref_c, _ = ref(x, x, x, attn_mask=~cm, need_weights=False)
    check("causal multi-head vs nn.MultiheadAttention", ours_c, ref_c, rows=rows)

    # Shape contract
    ok_shape = out.shape == x.shape
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] output shape equals input shape"
          f"{'':<21} {tuple(out.shape)}")
    rows.append(["output shape equals input shape", str(tuple(out.shape)),
                 "PASS" if ok_shape else "FAIL"])


def demo_gradient_flow(rows):
    heading("PART 6  GRADIENTS FLOW THROUGH THE WHOLE BLOCK")

    B, L, d_model, h = 1, 4, 32, 4
    torch.manual_seed(4)

    mha = MultiHeadAttention(d_model, h)
    x = torch.randn(B, L, d_model, requires_grad=True)

    out, _ = mha(x, x, x, mask=causal_mask(L))
    loss = out.pow(2).mean()
    loss.backward()

    print(f"\n  loss = {loss.item():.6f}")
    print(THIN)
    print(f"{'parameter':<22} | {'shape':>16} | {'grad norm':>12}")
    print(THIN)
    for name, p in mha.named_parameters():
        print(f"{name:<22} | {str(tuple(p.shape)):>16} | {p.grad.norm().item():>12.6f}")

    x_norm = x.grad.norm().item()
    print(f"{'input x':<22} | {str(tuple(x.shape)):>16} | {x_norm:>12.6f}")

    all_nonzero = all(p.grad.abs().sum().item() > 0 for p in mha.parameters())
    print(f"\n  [{'PASS' if all_nonzero else 'FAIL'}] every parameter received a non-zero gradient")
    print("  This is what makes the block trainable - nothing is detached and")
    print("  the -inf mask did not produce NaNs.")

    has_nan = any(torch.isnan(p.grad).any().item() for p in mha.parameters())
    print(f"  [{'PASS' if not has_nan else 'FAIL'}] no NaN gradients from the -inf mask")

    rows.append(["all params have non-zero grad", f"{x_norm:.6f}",
                 "PASS" if all_nonzero else "FAIL"])
    rows.append(["no NaN gradients", "0" if not has_nan else "found",
                 "PASS" if not has_nan else "FAIL"])


def demo_complexity(rows):
    heading("PART 7  WHY LONG CONTEXT IS EXPENSIVE")

    print("\nThe score matrix is L x L, so both memory and compute grow with L^2.")
    print("This single fact drives Flash Attention, sliding windows and every")
    print("other long-context trick.\n")

    print(f"{'seq len':>9} | {'score matrix':>16} | {'fp32 memory':>14} | {'vs L=512':>10}")
    print(THIN)

    base = None
    for L in (128, 512, 2048, 8192, 32768):
        cells = L * L
        mem_mb = cells * 4 / (1024 ** 2)
        if L == 512:
            base = mem_mb
        rel = f"{mem_mb / base:.0f}x" if base else "-"
        print(f"{L:>9} | {f'{L} x {L}':>16} | {mem_mb:>11.1f} MB | {rel:>10}")

    print("\n  Doubling the context quadruples the attention matrix. At 32k tokens")
    print("  a single head's scores need 4 GB in fp32 - per layer, per batch item.")
    rows.append(["L^2 memory at 32768", "4096.0 MB", "INFO"])


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------

def save_csv(rows, filename=CSV_FILENAME):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Check", "Value", "Result"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to '{filename}'")


def main():
    parser = argparse.ArgumentParser(
        description="Scaled dot-product and multi-head attention from scratch"
    )
    parser.add_argument("--seed", type=int, default=42, help="global torch seed")
    parser.add_argument("--no-csv", action="store_true", help="skip writing the CSV")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=4, sci_mode=False)

    print(RULE)
    print("SCALED DOT-PRODUCT AND MULTI-HEAD ATTENTION FROM SCRATCH")
    print(f"torch {torch.__version__}  |  seed {args.seed}  |  device cpu")
    print(RULE)

    rows = []
    demo_single_head(rows)
    demo_scaling_effect(rows)
    demo_causal_mask(rows)
    demo_padding_mask(rows)
    demo_multi_head(rows)
    demo_gradient_flow(rows)
    demo_complexity(rows)

    heading("SUMMARY")
    checks = [r for r in rows if r[2] in ("PASS", "FAIL")]
    passed = sum(1 for r in checks if r[2] == "PASS")
    print()
    for label, value, result in checks:
        print(f"  [{result}] {label:<48} {value}")
    print(THIN)
    print(f"  {passed}/{len(checks)} checks passed")

    if not args.no_csv:
        save_csv(rows)

    print(RULE)


if __name__ == "__main__":
    main()
