"""Offline GPU encoding with pinned local weights; no labels used for document vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer


def chunks_for(text, words=200, stride=150):
    tokens=text.split()
    for start in range(0,len(tokens),stride):
        yield ' '.join(tokens[start:start+words])
        if start+words>=len(tokens):break


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import torch
    from transformers import AutoModel
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModel.from_pretrained(args.model,local_files_only=True,trust_remote_code=False).eval().to(device)
    if device=='cuda':model.half()
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
    docs=json.loads((args.baseline/'doc_ids.json').read_text(encoding='utf-8'))
    texts=[];mapping=[]
    for i,d in enumerate(docs):
        cs=list(chunks_for((args.corpus/(d+'.txt')).read_text(encoding='utf-8')))
        texts.extend(cs);mapping.extend([i]*len(cs))
    mapping=np.asarray(mapping,np.uint16)
    assert np.array_equal(mapping,np.load(args.baseline/'chunk_doc.npy')),'Chunk alignment differs'
    queries=[json.loads(l)['question'] for l in (args.data/'sample-input.jsonl').read_text(encoding='utf-8').splitlines()]
    def encode(texts):
        result=[];start=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,len(texts),32):
                enc=tokenizer.encode_batch(texts[i:i+32])
                inputs={k:torch.tensor([getattr(e,attr) for e in enc],device=device) for k,attr in [('input_ids','ids'),('attention_mask','attention_mask'),('token_type_ids','type_ids')]}
                output=model(**inputs).last_hidden_state.float()
                mask=inputs['attention_mask'].unsqueeze(-1)
                vec=(output*mask).sum(1)/mask.sum(1).clamp(min=1)
                vec=torch.nn.functional.normalize(vec,p=2,dim=1)
                result.append(vec.cpu().numpy())
                if i%2048==0:print('encoded',i,'/',len(texts),'seconds',round(time.perf_counter()-start,1),flush=True)
        return np.concatenate(result)
    name=args.model.name
    vectors=encode(texts)
    np.save(args.output/(name+'_vectors.npy'),vectors)
    provenance={'model':json.loads((args.model/'source.json').read_text(encoding='utf-8')),
                'doc_ids_sha256':hashlib.sha256((args.baseline/'doc_ids.json').read_bytes()).hexdigest(),
                'chunk_doc_sha256':hashlib.sha256((args.baseline/'chunk_doc.npy').read_bytes()).hexdigest(),
                'vectors_sha256':hashlib.sha256((args.output/(name+'_vectors.npy')).read_bytes()).hexdigest(),
                'chunk_words':200,'chunk_stride':150,'document_tokens':256,
                'pooling':'attention-mask mean; L2 normalized','device':device,'torch':torch.__version__}
    (args.output/(name+'_vectors.meta.json')).write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    np.save(args.output/(name+'_queries.npy'),encode(queries))
    print('DONE',name,vectors.shape,flush=True)


if __name__=='__main__':main()
