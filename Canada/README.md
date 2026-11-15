# Canada — Two Solitudes

Train a bilingual linear classifier on the supplied synthetic corpus while suppressing synthetic style shortcuts.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python train.py /path/to/synthetic_reports.csv --output outputs
python /path/to/official/open_vault.py outputs/model.npz
```

## Input and output

The official synthetic corpus and open_vault.py. Use the original 4096-dimensional feature contract.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`featurize.py`](featurize.py)
- [`finalize.py`](finalize.py)
- [`requirements.txt`](requirements.txt)
- [`train.py`](train.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
