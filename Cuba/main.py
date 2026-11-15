"""Zemi: BM25 evidence and a local quantized semantic encoder. No runtime downloads."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np

STOPWORDS = set("a an the of and or to in on at by for from with as is are was were be been this that these those it its into than then there their they them he she his her we you not no but if so such which who what when where why how all any each more most other some very can could would should may might will do does did have has had about over under between through during before after".split())


def normalise(text):
    return re.sub(r"[^a-z0-9\s]", " ", "".join(c for c in
                  unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)))


def normalized(values):
    maximum = float(values.max(initial=0))
    return values / maximum if maximum > 0 else values


def char_ngrams(text):
    # sklearn char_wb: separate words, each padded with one space.
    for word in text.split()[:128]:
        word = ' ' + word[:100] + ' '
        for n in (3, 4, 5):
            if len(word) <= n:
                yield word
                break
            for i in range(len(word) - n + 1):
                yield word[i:i+n]


class Retriever:
    def __init__(self, index_dir: str, mode=None):
        p = Path(index_dir)
        self.config = json.loads((p / 'config.json').read_text(encoding='utf-8'))
        self.mode = mode or self.config['mode']
        if self.mode not in ('lexical', 'dense', 'hybrid'):
            raise ValueError('Unknown retrieval mode')
        manifest = json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256((p / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Index checksum mismatch: ' + name)
        with gzip.open(p / 'lexicon.json.gz', 'rt', encoding='utf-8') as stream:
            meta = json.load(stream)
        self.doc_ids, self.vocab = meta['doc_ids'], meta['vocab']
        self.stems, self.titles = meta['stems'], meta['titles']
        self.n = len(self.doc_ids)
        if self.n == 0 or len(set(self.doc_ids)) != self.n:
            raise ValueError('Document IDs must be nonempty and unique')
        load = lambda name: np.load(p / (name + '.npy'), allow_pickle=False)
        self.cd = load('chunk_doc')
        self.ptr, self.ids = load('ptr'), load('ids')
        self.values = load('values').astype(np.float32)
        if len(self.ptr) != max(self.vocab.values()) + 2 or self.ptr[-1] != len(self.ids) or len(self.ids) != len(self.values):
            raise ValueError('Sparse index dimensions differ')
        if np.any(self.ptr[1:] < self.ptr[:-1]) or self.ids.max(initial=0) >= len(self.cd) or self.cd.max(initial=0) >= self.n:
            raise ValueError('Sparse index IDs or pointers invalid')
        if not np.isfinite(self.values).all():
            raise ValueError('Nonfinite lexical weights')
        df = np.diff(self.ptr).astype(np.float32)
        self.idf = np.log1p((len(self.cd) - df + .5) / (df + .5))
        self.correction_weight = self.config['correction_weight']
        self.dense_calls = 0
        self.vectors = self.session = self.tokenizer = None
        if self.mode != 'lexical':
            import onnxruntime as ort
            from tokenizers import Tokenizer
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = 1
            opt.inter_op_num_threads = 1
            opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(str(p / 'encoder.onnx'), sess_options=opt, providers=['CPUExecutionProvider'])
            self.tokenizer = Tokenizer.from_file(str(p / 'tokenizer.json'))
            self.tokenizer.no_padding()
            self.tokenizer.enable_truncation(max_length=self.config['query_tokens'])
            quant, scales = load('vectors_i8'), load('vector_scales').astype(np.float32)
            if quant.shape != (len(self.cd), self.config['dimension']) or scales.shape != (len(self.cd),):
                raise ValueError('Dense vector dimensions or chunk alignment differ')
            self.vectors = quant.astype(np.float32) * scales[:, None]
            if not np.isfinite(self.vectors).all() or np.any(scales <= 0):
                raise ValueError('Invalid dense vectors')
            self._query_vector('encoder initialization check')
        self.char = None
        if self.config.get('char_weight', 0) and self.mode != 'dense':
            with gzip.open(p / 'char_vocab.json.gz', 'rt', encoding='utf-8') as stream:
                self.char_vocab = json.load(stream)
            self.char_ptr, self.char_ids = load('char_ptr'), load('char_ids')
            self.char_values, self.char_idf = load('char_values').astype(np.float32), load('char_idf').astype(np.float32)
            if len(self.char_ptr) != len(self.char_idf) + 1 or self.char_ptr[-1] != len(self.char_ids) or len(self.char_ids) != len(self.char_values):
                raise ValueError('Character index dimensions differ')
            self.char = True

    def _query_vector(self, text):
        e = self.tokenizer.encode(text)
        inputs = {key: np.array([getattr(e, attr)], np.int64) for key, attr in
                  [('input_ids', 'ids'), ('attention_mask', 'attention_mask'), ('token_type_ids', 'type_ids')]}
        hidden = self.session.run(None, inputs)[0][0]
        mask = np.array(e.attention_mask, np.float32)
        vec = (hidden * mask[:, None]).sum(0) / max(float(mask.sum()), 1)
        if vec.shape != (self.config['dimension'],) or not np.isfinite(vec).all():
            raise ValueError('Encoder returned an invalid vector')
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _correct(self, word):
        if not 5 <= len(word) <= 32 or not word.isalpha():
            return None
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        variants = {a + b[1:] for a, b in splits if b}
        variants.update(a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
        variants.update(a + c + b[1:] for a, b in splits if b for c in alphabet)
        variants.update(a + c + b for a, b in splits for c in alphabet)
        candidates = [(int(self.ptr[t + 1] - self.ptr[t]), w)
                      for w in variants if (t := self.vocab.get(w)) is not None]
        return max(candidates)[1] if candidates else None

    def _lexical(self, text):
        words = [w for w in text.split()[:128] if w not in STOPWORDS]
        weights, counts = {}, {}
        corrected = 0
        for word in words:
            weight = 1.
            if word not in self.vocab and self.correction_weight and corrected < 4:
                corrected += 1
                word, weight = self._correct(word), self.correction_weight
            if word in self.vocab:
                tid = self.vocab[word]
                weights[tid] = max(weights.get(tid, 0.), weight)
                counts[tid] = min(2., counts.get(tid, 0.) + weight)
        chunks, coverage = np.zeros(len(self.cd), np.float32), np.zeros(self.n, np.float32)
        for tid in sorted(weights):
            a, b = self.ptr[tid:tid+2]
            ids, values = self.ids[a:b], self.values[a:b]
            chunks[ids] += values * counts[tid]
            # Reconstruct document evidence; no duplicate global arrays on disk.
            pool = np.zeros(self.n, np.float32)
            np.maximum.at(pool, self.cd[ids], values / (self.idf[tid] + 1e-9))
            coverage += pool * weights[tid]
        passage = np.zeros(self.n, np.float32)
        np.maximum.at(passage, self.cd, chunks)
        title = np.zeros(self.n, np.float32)
        for stem in sorted({self.stems.get(w, w) for w in re.findall('[a-z]+', text)}):
            if stem in self.titles:
                ids, weight = self.titles[stem]
                title[ids] += weight
        return (self.config['passage_weight'] * normalized(passage) +
                self.config['coverage_weight'] * normalized(coverage) +
                self.config['title_weight'] * normalized(title))

    def _characters(self, text):
        counts = Counter(char_ngrams(text))
        q = [(self.char_vocab[g], count) for g, count in sorted(counts.items()) if g in self.char_vocab]
        scores = np.zeros(self.n, np.float32)
        if q:
            ids = np.array([i for i, _ in q])
            weights = (1 + np.log([c for _, c in q])) * self.char_idf[ids]
            weights /= np.linalg.norm(weights) + 1e-9
            for tid, weight in zip(ids, weights):
                a, b = self.char_ptr[tid:tid+2]
                scores[self.char_ids[a:b]] += self.char_values[a:b] * weight
        return scores

    def search(self, query: str, k: int = 3) -> list[str]:
        if k <= 0:
            return []
        query = query[:4096]
        if not query.strip():
            return self.doc_ids[:min(k, self.n)]
        text = normalise(query)
        score = self._lexical(text) if self.mode != 'dense' else np.zeros(self.n, np.float32)
        if self.mode != 'lexical':
            q = self._query_vector(query)
            self.dense_calls += 1
            dense = np.zeros(self.n, np.float32)
            np.maximum.at(dense, self.cd, self.vectors @ q)
            score += self.config['dense_weight'] * normalized(dense)
        if self.char:
            score += self.config['char_weight'] * normalized(self._characters(text))
        best = np.argsort(-score, kind='stable')[:min(k, self.n)]
        return [self.doc_ids[int(i)] for i in best]
