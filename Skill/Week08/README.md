# GenAI — CO-2 Week08

**Name:** Ramagiri Rishik
**Roll No:** 2520030333
**Subject:** GenAI
**Course Outcome:** CO-2 — Week08

## Contents

| File | Description |
| --- | --- |
| `attention_from_scratch.py` | Scaled dot-product attention and multi-head attention implemented from first principles in PyTorch |
| `attention_results.csv` | Check-by-check results written by the script |
| `output.txt` | Captured console output of a full run |
| `requirements.txt` | Python dependencies (`torch`) |

## About the Lab

The script builds attention from scratch and verifies every tensor against PyTorch's own
reference implementations (`F.scaled_dot_product_attention` and `nn.MultiheadAttention`):

1. **Scaled dot-product attention** — raw scores, scaling, softmax, weighted sum of values.
2. **Why divide by `sqrt(d_k)`** — score variance grows with `d_k` and saturates softmax without scaling.
3. **Causal mask** — the upper triangle is forbidden, so a token cannot see the future.
4. **Padding mask** — padded key positions receive exactly zero weight in a batch.
5. **Multi-head attention** — heads split `d_model`, attend differently, and are recombined by an output projection.
6. **Gradient flow** — every parameter receives a non-zero gradient and the `-inf` mask produces no NaNs.
7. **`O(L²)` cost** — the score matrix grows quadratically with sequence length.

## Setup and Run

```bash
pip install -r requirements.txt
python attention_from_scratch.py
```

Optional flags:

```bash
python attention_from_scratch.py --seed 7     # different random seed
python attention_from_scratch.py --no-csv     # skip writing attention_results.csv
```

## Expected Result

A full run prints each demonstration and ends with a summary of **10/10 checks passed**,
then writes 16 rows to `attention_results.csv`.
