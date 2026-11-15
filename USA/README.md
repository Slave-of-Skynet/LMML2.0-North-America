# United States — The Great American Token Mystery

Identify the tokenizer from exact calibration encodings and decode the payload with that tokenizer.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py /path/to/test-input.json outputs/predictions.json
```

## Input and output

The organizer tokenizer-calibration and payload JSON; public tokenizer files are fetched unless --offline is used.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)
- [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
