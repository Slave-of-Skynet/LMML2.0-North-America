"""Build the complete data-only Zemi index from pinned source assets, without labels."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import Stemmer

from main import normalise


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gzip(path, value):
    # No timestamp or filename in the header, so a rebuild is byte reproducible.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def build(source, metadata, destination, model=None, vectors=None, corpus=None, char_cache=None):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new empty index directory; stale assets are not reused')
    destination.mkdir(parents=True, exist_ok=True)
    docs = json.loads((source/'doc_ids.json').read_text(encoding='utf-8'))
    cd = np.load(source/'chunk_doc.npy', allow_pickle=False)
    base = source/'bm25'
    ptr = np.load(base/'indptr.csc.index.npy', allow_pickle=False)
    ids = np.load(base/'indices.csc.index.npy', allow_pickle=False)
    values = np.load(base/'data.csc.index.npy', allow_pickle=False)
    vocab = json.loads((base/'vocab.index.json').read_text(encoding='utf-8'))
    vocab = {w:t for w,t in vocab.items() if t+1<len(ptr)}
    if len(cd)>=65536 or len(docs)>=65536 or len(set(docs))!=len(docs):
        raise ValueError('Index cannot use uint16 IDs')
    for name, array in [('chunk_doc',cd.astype(np.uint16)),('ptr',ptr.astype(np.uint32)),
                        ('ids',ids.astype(np.uint16)),('values',values.astype(np.float16))]:
        np.save(destination/(name+'.npy'),array)
    stemmer = Stemmer.Stemmer('english')
    meta = {r['id']:r for r in map(json.loads,Path(metadata).read_text(encoding='utf-8').splitlines())}
    if set(docs)!=set(meta):raise ValueError('Baseline IDs and metadata differ')
    titles=defaultdict(list)
    stop=set('chapter part cuba cuban i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi'.split())
    stems=dict(zip(vocab,stemmer.stemWords(list(vocab))))
    for did,d in enumerate(docs):
        tokens=re.findall('[a-z]+',normalise(meta[d]['title']))
        stems.update(zip(tokens,stemmer.stemWords(tokens)))
        for word in sorted(set(stemmer.stemWords(tokens))-stop):titles[word].append(did)
    titles={w:[posting,math.log1p(len(docs)/(len(posting)+1))] for w,posting in titles.items()}
    write_gzip(destination/'lexicon.json.gz',{'doc_ids':docs,'vocab':vocab,'stems':stems,'titles':titles})
    config={'mode':'hybrid' if model else 'lexical','dimension':384,'query_tokens':128,
            'passage_weight':.5,'coverage_weight':.4,'title_weight':.2,'dense_weight':1.,
            'correction_weight':.35,'char_weight':.15 if corpus or char_cache else 0.}
    provenance={'baseline_doc_ids_sha256':sha256(source/'doc_ids.json'),
                'baseline_chunk_doc_sha256':sha256(source/'chunk_doc.npy'),
                'metadata_sha256':sha256(metadata)}
    if model:
        model,vectors=Path(model),Path(vectors)
        encoded=json.loads(vectors.with_suffix('.meta.json').read_text(encoding='utf-8'))
        if encoded['doc_ids_sha256']!=provenance['baseline_doc_ids_sha256'] or encoded['chunk_doc_sha256']!=provenance['baseline_chunk_doc_sha256']:
            raise ValueError('Encoded vector order does not match baseline')
        if encoded['vectors_sha256']!=sha256(vectors):raise ValueError('Encoded vectors checksum differs')
        if encoded['model']!=json.loads((model/'source.json').read_text(encoding='utf-8')):raise ValueError('Encoder and vectors use different model revisions')
        dense=np.load(vectors,allow_pickle=False)
        if dense.shape!=(len(cd),384) or not np.isfinite(dense).all():raise ValueError('Invalid dense vectors')
        scales=np.maximum(np.abs(dense).max(1),1e-8)/127
        quant=np.clip(np.rint(dense/scales[:,None]),-127,127).astype(np.int8)
        np.save(destination/'vectors_i8.npy',quant)
        np.save(destination/'vector_scales.npy',scales.astype(np.float16))
        shutil.copyfile(model/'onnx/model_quint8_avx2.onnx',destination/'encoder.onnx')
        shutil.copyfile(model/'tokenizer.json',destination/'tokenizer.json')
        provenance['dense_encoding']=encoded
        shutil.copyfile(Path(__file__).with_name('MODEL_LICENSE.txt'),destination/'MODEL_LICENSE.txt')
        (destination/'MODEL_SOURCE.txt').write_text(
            'Quantized multi-qa-MiniLM-L6-cos-v1; Apache-2.0.\n'
            'https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1\n'
            'Revision: '+encoded['model']['revision']+'\n',encoding='utf-8',newline='\n')
    if corpus or char_cache:
        from scipy import sparse
        if char_cache:
            cache=Path(char_cache)
            matrix=sparse.load_npz(cache/'char_matrix.npz').tocsr()
            cvocab=json.loads((cache/'char_vocab.json').read_text(encoding='utf-8'))
            idf=np.load(cache/'char_idf.npy',allow_pickle=False)
            # Cache is an optional local acceleration; clean reproduction fits below.
            cached_docs=json.loads((cache/'diagnosis.json').read_text(encoding='utf-8'))['doc_ids']
            if matrix.shape[0]!=len(docs) or cached_docs!=docs:raise ValueError('Character cache document order differs')
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts=[normalise((Path(corpus)/(d+'.txt')).read_text(encoding='utf-8')) for d in docs]
            cv=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_df=.8,max_features=100000,sublinear_tf=True,dtype=np.float32)
            matrix=cv.fit_transform(texts).tocsr();cvocab=cv.vocabulary_;idf=cv.idf_
        # Keep the 250 strongest grams per document, preserving original L2 norms.
        for i in range(matrix.shape[0]):
            a,b=matrix.indptr[i:i+2]
            if b-a>250:
                keep=np.argpartition(matrix.data[a:b],-250)[-250:]
                mask=np.ones(b-a,bool);mask[keep]=False;matrix.data[a:b][mask]=0
        matrix.eliminate_zeros();matrix=matrix.tocsc();matrix.sort_indices()
        for name,array in [('char_ptr',matrix.indptr.astype(np.uint32)),('char_ids',matrix.indices.astype(np.uint16)),
                           ('char_values',matrix.data.astype(np.float16)),('char_idf',idf.astype(np.float16))]:
            np.save(destination/(name+'.npy'),array)
        write_gzip(destination/'char_vocab.json.gz',{k:int(v) for k,v in cvocab.items()})
    (destination/'config.json').write_text(json.dumps(config,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    manifest={'provenance':provenance,'sha256':{p.name:sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    (destination/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Index bytes:',sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline_index');ap.add_argument('metadata');ap.add_argument('output_index')
    ap.add_argument('--model',type=Path);ap.add_argument('--vectors',type=Path)
    ap.add_argument('--corpus',type=Path);ap.add_argument('--char-cache',type=Path)
    args=ap.parse_args()
    if bool(args.model)!=bool(args.vectors):ap.error('--model and --vectors must be supplied together')
    build(args.baseline_index,args.metadata,args.output_index,args.model,args.vectors,args.corpus,args.char_cache)
