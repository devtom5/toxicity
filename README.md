# Toxicity Classification + LD50 Prediction

End-to-end ML project for:
- Binary classification: toxic vs non-toxic
- Regression: LD50 prediction
- Feature generation using PaDEL + Mordred (Mordred enabled by default)
- Feature selection by importance
- API (FastAPI) and Web demo (Streamlit)

## Quickstart

1. Install deps:

```bash
python3 -m pip install -r requirements.txt
```

2. Put your raw file at:

```
/Users/devaasirvatham/Documents/New project/data/raw_compound.csv
```

3. Generate features and split:

```bash
python3 -m src.toxicity.prepare_data --config config.yaml
```

4. Train models:

```bash
python3 -m src.toxicity.train --config config.yaml
```

5. Evaluate:

```bash
python3 -m src.toxicity.evaluate --config config.yaml
```

6. Run API:

```bash
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

7. Run Web demo:

```bash
python3 -m streamlit run web/app.py --server.address 127.0.0.1 --server.port 8501
```

## Data Expectations

The raw CSV should include:
- `SMILES`
- `Toxicity` (0/1)
- `Toxicity Value` (LD50)

## Feature Generation

Configured in `config.yaml`:
- `feature_generation.use_padel`: set to `true` to enable PaDEL descriptors
- `feature_generation.use_mordred`: enable Mordred descriptors
- `feature_generation.mordred_n`: limit number of Mordred descriptors for speed

PaDEL can be slower; if it hangs, keep it disabled and increase Mordred descriptors instead.

## Outputs

- Models saved in `models/`
- Feature splits in `data/`
- Metadata in `models/featuregen_meta.json`
