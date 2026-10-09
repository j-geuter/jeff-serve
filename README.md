# Jeff: Jev-style decision models from open LLMs

Serving code for **Jeff-1.0-Large**, a Jev-style decision model: a state, a typed question, and its options go in, and
one forward pass returns a probability for each option, all in a single forward pass.

- Model weights: [jgeuter/Jeff-1.0-Large](https://huggingface.co/jgeuter/Jeff-1.0-Large) (Gemma 4 31B + LoRA, merged)

## How it works

Each question is rendered as a multiple-choice prompt. A decision head turns the last hidden state $h$ into one
logit per option, and a softmax over the $K$ options gives the probabilities:

$$z = c\tanh(W_0 h / c) + A \mathrm{std}(h) + b, \qquad p = \mathrm{softmax}(z_1, \dots, z_K),$$

where $W_0$ are the language-model-head rows of the answer letters (A, B, C, ...), $c = 30$ is Gemma's final-logit
soft-capping, $\mathrm{std}(h)$ standardizes each feature with statistics computed once on training data, and the
learned correction $A$, $b$ starts at zero, so training starts exactly at the model's own letter readout.

Question types: `choice` (2 to 16 options), `noul` (yes/no; the answer is $P(\text{yes})$) and `score` (ordered levels,
2 to 16). Inputs up to 8,192 tokens.

## Run with Docker

```bash
# 1. Fetch the weights (pin the revision you want; about 62 GB).
huggingface-cli download jgeuter/Jeff-1.0-Large --local-dir /models/Jeff-1.0-Large
# 2. Build and serve (one GPU with at least 80 GB, e.g. H100 80GB, H200 or RTX PRO 6000 96GB).
docker build -t jeff .
docker run --gpus '"device=0"' -v /models/Jeff-1.0-Large:/model:ro -p 8013:8013 jeff
# 3. Check it.
python scripts/smoke_test.py --endpoint http://127.0.0.1:8013
```

The container reads the weights from `/model` and needs no network access.

## Run without Docker

```bash
pip install -r requirements.txt && pip install .
jeff-serve /models/Jeff-1.0-Large --host 127.0.0.1 --port 8013     # or the repo id jgeuter/Jeff-1.0-Large
```

Tested with Python 3.12, torch 2.13.0 (CUDA 13.0), transformers 5.17.0 and accelerate 1.15.0 (`requirements.txt`).

## Request/response format

```
POST /v1/systemone
{"state": {...} or "text",
 "model": "Jeff-1.0-Large",
 "questions": {
   "route":    {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"}},
   "urgent":   {"type": "noul", "instructions": "Does this need a reply today?",
                "criteria": {"true": "Reply today", "false": "Can wait"}},
   "severity": {"type": "score", "instructions": "How severe is the problem?",
                "criteria": ["trivial", "minor", "moderate", "serious", "critical"]}}}
->
{"answers": {"route":    {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "technical": 0.03}},
             "urgent":   {"type": "noul", "noul": 0.801},
             "severity": {"type": "score", "score": 2, "probabilities": {"0": 0.01, "1": 0.2, ...}}},
 "model": "Jeff-1.0-Large", "usage": {"input_tokens": 612, "output_tokens": 0}, "runtime": {...}}
```

Each question is one forward pass; questions of one request are answered one after another.

## Python

```python
from jeff import JeffModel
model = JeffModel("/models/Jeff-1.0-Large")
model.decide({"ticket": "Refund not received after 14 days"},
             {"route": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
```

## Licence

The code is Apache-2.0 (`LICENSE`); it contains code derived from SemIf and JevBench (MIT), see `NOTICE`. The model
weights have their own licence, given in the model card.
