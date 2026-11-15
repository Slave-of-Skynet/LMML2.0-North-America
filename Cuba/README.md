# Cuba — If you know, you know!

Combine BM25 passage/title evidence, quantized semantic retrieval and a bounded character n-gram correction signal.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
# Download and unpack the ready-to-evaluate release artifact linked below.
# Runtime API: Retriever(index_dir).search(query, k).
python evaluate.py --help
```

## Input and output

For runtime, the ready-to-evaluate release ZIP. For rebuilding, all 1552 corpus texts, corpus_meta.jsonl, baseline index and pinned encoder assets.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/13m0PfjC563EcC3DsNeADWCZ21Vm8qb3j).

## Files

- [`build_index.py`](build_index.py)
- [`download_model.py`](download_model.py)
- [`encode_transformer.py`](encode_transformer.py)
- [`evaluate.py`](evaluate.py)
- [`main.py`](main.py)
- [`MODEL_LICENSE.txt`](MODEL_LICENSE.txt)
- [`package_submission.py`](package_submission.py)
- [`requirements-build.txt`](requirements-build.txt)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

## Ready-to-evaluate artifact

Download [zemi_recovery_v3_1.zip](https://github.com/Slave-of-Skynet/LMML2.0-North-America/releases/download/jury-artifacts-20261115/zemi_recovery_v3_1.zip).
SHA-256: `e41562ae2243b86cc510157ece2a937ec18e30b9dae826bc2409a753fe4fcf20`.
It contains `main.py` and its complete data-only `index/`, including the quantized local encoder.
The archive includes model provenance and the Apache-2.0 license. Runtime needs no downloads.

To rebuild, install `requirements-build.txt` in a separate environment and obtain the complete
official corpus, corpus metadata and baseline index. Run from this directory:

```bash
python download_model.py outputs/model
python encode_transformer.py --model outputs/model --corpus /path/to/corpus --baseline /path/to/baseline/index --data /path/to/task/data --output outputs
python build_index.py /path/to/baseline/index /path/to/task/data/corpus_meta.jsonl outputs/rebuilt/index --model outputs/model --vectors outputs/multi-qa-MiniLM-L6-cos-v1_vectors.npy --corpus /path/to/corpus
python package_submission.py outputs/rebuilt/index outputs/rebuilt.zip
```

The base encoder revision is `b207367332321f8e44f96e224ef15bc607f4dbf0`.
Use `requirements.txt` for the isolated evaluator runtime and the organizer's original checker
for its official limits. `evaluate.py` is an additional local evaluator, not the official checker.
