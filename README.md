# Hack4Delhi: Government Spending Fraud Transformer (prototype)

> **Status: early prototype, not runnable as-is.** This is a single-commit scaffold from January 2026 (`adfe298`). It sketches the pipeline but has not been trained or evaluated: there are no results, no saved model and no data in the repo. The known blockers are listed under [Limitations](#limitations--next-steps). The fuller team build of this idea, with a trained model, backend and dashboard, is [VajraAI](https://github.com/aadvikchaturvedi/VajraAI).

Solo project by Aadvik Chaturvedi.

## Idea

Public procurement records in India are rarely labelled for fraud, but the General Financial Rules (GFR) and CVC guidelines describe well-known red flags. The plan here is to:

1. pull procurement records from the Open Government Data platform (data.gov.in),
2. turn those red flags into weak labels, and
3. train a small tabular Transformer on those labels so it can score new transactions, including ones that match no single rule exactly.

## Pipeline

```
data.gov.in API ──> IndiaDataIngestor ──> apply_indian_red_flags ──> CSV
 (resource JSON)     src/ingestion.py      src/labelling.py          data/processed_india_spending.csv
                                                                          │
                     TransformerFraudModel <── GovSpendingDataset <───────┘
                     src/model.py              src/dataset.py
                          │
                     train_model (BCE, Adam 1e-4) ──> models/transformer_fraud_v1.pth
                     src/trainer.py
```

### Ingestion (`src/ingestion.py`)

`IndiaDataIngestor.fetch_resource(resource_id, limit)` calls `https://api.data.gov.in/resource/<id>` with an API key and returns the `records` array as a DataFrame. `main.py` asks for 5,000 records.

### Heuristic labels (`src/labelling.py`)

| Flag | Rule |
|---|---|
| `is_split_purchase` | Amount between ₹4.5 lakh and ₹5 lakh, i.e. just under the ₹5 lakh threshold, a common sign of a purchase split to avoid a tender |
| `is_single_bid` | `bidder_count == 1` in a competitive tender (many states require 3 bids) |
| `repeat_win_streak` | Number of contracts for each (department, vendor) pair. Computed but not yet used in the label |
| `heuristic_label` | `is_split_purchase OR is_single_bid` |

`config/config.yaml` also defines `split_purchase_limit: 500000`, `high_risk_z_score: 3.0` and `contamination_rate: 0.01`, but no code reads the config yet.

### Model (`src/model.py`)

`TransformerFraudModel` is a feature-tokenising Transformer, with one token per input feature:

- each categorical column (`dept_id`, `vendor_id`, `procurement_type`) gets its own `nn.Embedding`,
- the continuous columns (`amount`, `days_since_last_contract`, `z_score_dept`) go through one linear layer, which is reshaped into one token per feature,
- a 3-layer `nn.TransformerEncoder` (embed dim 64, 4 heads, FFN 4×) runs over the 6 tokens,
- the tokens are mean-pooled and passed through a sigmoid output.

### Training (`src/trainer.py`, `main.py`)

BCE loss, Adam at lr 1e-4, 20 epochs, batch size 32. The state dict is saved to `models/transformer_fraud_v1.pth`. `src/utils.py` holds a second, shorter entry point that trains straight from `data/raw_spending.csv` and skips ingestion.

## Layout

```
govt_fraud_detection/
├── main.py             end-to-end: ingest -> label -> train -> save
├── config/config.yaml  thresholds and model hyperparameters (not loaded yet)
├── requirements.txt
└── src/
    ├── ingestion.py    data.gov.in client
    ├── labelling.py    GFR/CVC red-flag rules
    ├── dataset.py      scaling + label encoding -> torch Dataset
    ├── model.py        TransformerFraudModel
    ├── trainer.py      training loop, get_anomaly_score()
    └── utils.py        alternate training script
```

## Intended usage

Once the blockers below are fixed:

```bash
cd govt_fraud_detection
pip install -r requirements.txt
export OGD_INDIA_API_KEY=<your data.gov.in key>
# set RESOURCE_ID in main.py to a procurement dataset id from data.gov.in
mkdir -p data
python main.py
```

## Limitations / next steps

The code compiles, but `main.py` will fail at runtime. Here is what has to change before it runs, in pipeline order:

- `main.py` imports `src.labeling`, but the file is `src/labelling.py`.
- `RESOURCE_ID` is a truncated placeholder (`"f507b5e3-9977-463d-..."`), and `data/` is not created before the CSV is written.
- The labeller expects `amount`, `bidder_count`, `dept_id`, `vendor_id` and `id` columns. Raw data.gov.in records use their own field names, and there is no mapping or cleaning step yet.
- `days_since_last_contract` and `z_score_dept` are read by the dataset but computed nowhere. The z-score > 3 rule exists only in the config.
- The labeller writes `heuristic_label`, but `GovSpendingDataset` reads `is_fraud`.
- `src/dataset.py` uses `np` without importing NumPy, and `src/trainer.py` uses `nn` without importing `torch.nn`.
- `config.yaml` is never loaded, and `main.py` hard-codes its own values (embed dim 64, not the config's 128).
- `rtdl` and `xgboost` are listed in `requirements.txt` but not used.

Modelling gaps once it runs:

- The model is trained to reproduce two hand-written rules, so on its own it cannot learn anything beyond them. Next steps are to evaluate it against the rules on held-out data, add the z-score and repeat-vendor signals as features rather than labels, and compare against an unsupervised baseline such as Isolation Forest using `contamination_rate`.
- `get_anomaly_score` returns the classifier's sigmoid output. Despite its docstring, it does not use attention weights.
- There is no train/validation split and no metric is computed.
