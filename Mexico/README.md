# Mexico — Ruta de las Identidades

Perform deterministic record linkage using normalized identity fields and local consistency rules.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py /path/to/test-input.csv outputs/predictions.json
```

## Input and output

The organizer identity-record CSV.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
