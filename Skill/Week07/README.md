# Next-Token Probability Lab

**Name:** Ramagiri Rishik  
**Roll Number:** 2520030333  
**Subject:** GenAI  
**Course Outcome:** CO-2  
**Week:** Week07

## Description

This lab demonstrates next-token prediction using the open-source GPT-2 model.
It accepts a text prompt, tokenizes the input, calculates raw logits, applies
temperature scaling and softmax, and displays the top-k candidate tokens with
their probabilities.

## Files

- `next_token_prob.py` - Main interactive Python program
- `requirement.txt` - Required Python packages
- `gpt2_next_token_results.csv` - Saved prediction results
- `output.txt` - Example program output

## Requirements

- Python 3
- PyTorch
- Transformers

Install the packages with:

```powershell
python -m pip install -r requirement.txt
```

## Run

From this directory, run:

```powershell
python next_token_prob.py
```

Enter a prompt, top-k value, and temperature at the prompts. Type `exit`,
`quit`, or `q` to stop the program.
