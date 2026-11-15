"""Remove synthetic boilerplate directions from a trained linear classifier.

The mask uses only the supplied synthetic corpus and a fixed bilingual list
of function/context words. It never reads sealed fingerprints or vault data.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from featurize import bucket, tokens, DIM

STOP = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after 0 un une le la les de du des au aux et ou en dans sur par pour avec est sont était étaient être ce cet cette ces que qui quand où se sa son ses leur leurs il elle ils elles ont eu nous vous ne pas plus près environ plusieurs quelque dont many large severe strong several near north south east west northern southern eastern western region area ville région town city personnes people reported estimated damage damages destroyed homes maisons buildings".split())


def nuisance_mask(texts):
    mask = np.ones(DIM)
    for word in STOP:
        mask[bucket(word)] = 0
    for text in texts:
        words = tokens(text)
        for a, b in zip(words, words[1:]):
            if a in STOP and b in STOP:
                mask[bucket(a + " " + b)] = 0
    return mask


def finalize_model(base, corpus, output):
    with open(corpus, encoding="utf-8", newline="") as stream:
        mask = nuisance_mask(row["text"] for row in csv.DictReader(stream))
    with np.load(base, allow_pickle=False) as checkpoint:
        W = np.asarray(checkpoint["W"], dtype=np.float64) * mask
    if W.shape != (9, DIM) or not np.isfinite(W).all():
        raise ValueError("Invalid coefficient matrix")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, W=W, b=np.zeros(9, dtype=np.float64))
    return int(mask.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model")
    parser.add_argument("corpus")
    parser.add_argument("output")
    args = parser.parse_args()
    print("Active feature cells:", finalize_model(args.base_model, args.corpus, args.output))
