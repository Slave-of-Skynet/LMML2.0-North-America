# Panama — The Canal Knows Too Much

Audit benchmark contamination and produce supported decisions from the supplied evidence bundle.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python audit.py /path/to/official_bundle --output outputs/report.json --evidence outputs/decisions.json
```

## Input and output

The supplied audit bundle, including benchmark and retrieval evidence.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`audit.py`](audit.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
