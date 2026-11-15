"""Measure an extracted ZIP in an isolated, offline subprocess; never tune here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import zipfile

RUNNER = r'''
import json,sys,time,socket,hashlib
from pathlib import Path
def no_network(*a,**k):raise OSError('network disabled')
socket.create_connection=no_network;socket.getaddrinfo=no_network
class OfflineSocket(socket.socket):
 def connect(self,*a,**k):no_network()
 def connect_ex(self,*a,**k):no_network()
socket.socket=OfflineSocket
root,questions,output,mode=sys.argv[1:]
import os;os.chdir(root);sys.path.insert(0,root)
t=time.perf_counter()
from main import Retriever
r=Retriever('index',mode=mode)
load=time.perf_counter()-t
qs=json.loads(Path(questions).read_text(encoding='utf-8'))
valid=set(r.doc_ids)
for q in qs[:10]:r.search(q['question'],3)
results=[]
for q in qs:
 t=time.perf_counter();ids=r.search(q['question'],3);lat=(time.perf_counter()-t)*1000
 assert len(ids)==3 and len(set(ids))==3 and set(ids)<=valid
 results.append({'qid':q['qid'],'ids':ids,'ms':lat})
stress=[]
for q in ['', 'x'*100000, 'institushuns Soviot rnilliliter '*500, 'history '*10000, 'José Martí — Guantánamo '*1000]:
 t=time.perf_counter();ids=r.search(q,3);stress.append((time.perf_counter()-t)*1000)
 assert len(ids)==3 and len(set(ids))==3 and set(ids)<=valid
assert r.search('test',0)==[]
assert len(r.search('test',len(valid)+1))==len(valid)
assert r.search('Cuba')==r.search('Cuba')
assert (r.dense_calls>0)==(mode!='lexical')
Path(output).write_text(json.dumps({'load_s':load,'results':results,'stress_ms':stress,'dense_calls':r.dense_calls}),encoding='utf-8')
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--modes',nargs='+',default=['lexical','dense','hybrid'])
    ap.add_argument('--synthetic-ocr',action='store_true');ap.add_argument('--repeat',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    queries=[json.loads(l) for l in (args.data/'sample-input.jsonl').read_text(encoding='utf-8').splitlines()]
    gold={r['qid']:r['doc_id'] for r in map(json.loads,(args.data/'sample-output.jsonl').read_text(encoding='utf-8').splitlines())}
    meta={r['id']:r for r in map(json.loads,(args.data/'corpus_meta.jsonl').read_text(encoding='utf-8').splitlines())}
    if args.synthetic_ocr:
        rng=random.Random(20261115)
        def corrupt(text):
            def alter(m):
                word=m[0]
                if len(word)<5 or rng.random()>.3:return word
                i=rng.randrange(len(word)-1)
                return word[:i]+word[i+1]+word[i]+word[i+2:]
            return re.sub('[A-Za-z]+',alter,text)
        queries=[dict(q,question=corrupt(q['question'])) for q in queries]
    all_results={}
    with tempfile.TemporaryDirectory(prefix='cuba_evaluate_') as directory:
        root=Path(directory);sub=root/'submission'
        with zipfile.ZipFile(args.archive) as archive:
            if archive.testzip() is not None:raise ValueError('CRC error')
            archive.extractall(sub)
            unpacked=sum(i.file_size for i in archive.infolist())
        runner=root/'runner.py';runner.write_text(RUNNER,encoding='utf-8')
        qfile=root/'queries.json';qfile.write_text(json.dumps(queries),encoding='utf-8')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        for mode in args.modes+(['hybrid-repeat'] if args.repeat else []):
            out=root/(mode+'.json')
            subprocess.run([sys.executable,'-I',str(runner),str(sub),str(qfile),str(out),mode.split('-')[0]],check=True,env=env,cwd=root,timeout=300)
            res=json.loads(out.read_text(encoding='utf-8'));rows=res['results']
            def metric(selected,k):return sum(gold[r['qid']] in r['ids'][:k] for r in selected)/max(1,len(selected))
            dev=[];validation=[];ocr=[];other=[]
            for r in rows:
                group=int(hashlib.sha256(('cuba-recovery-20261115:'+gold[r['qid']]).encode()).hexdigest()[:8],16)%5
                (validation if group==0 else dev).append(r)
                (ocr if meta[gold[r['qid']]]['kind']=='ocr_chapter' else other).append(r)
            lat=sorted(r['ms'] for r in rows)
            summary={'hit1':metric(rows,1),'hit3':metric(rows,3),'hits3':sum(gold[r['qid']] in r['ids'] for r in rows),'n':len(rows),
                     'dev_hit3':metric(dev,3),'validation_hit3':metric(validation,3),'validation_n':len(validation),
                     'ocr_source_hit3':metric(ocr,3),'ocr_source_n':len(ocr),'other_source_hit3':metric(other,3),
                     'load_s':res['load_s'],'p50_ms':lat[round(.5*(len(lat)-1))],'p95_ms':lat[round(.95*(len(lat)-1))],
                     'max_ms':max(lat),'stress_max_ms':max(res['stress_ms']),'unpacked_bytes':unpacked,
                     'dense_calls':res['dense_calls'],'ids_sha256':hashlib.sha256(json.dumps([r['ids'] for r in rows]).encode()).hexdigest()}
            all_results[mode]={'summary':summary,**res}
            print(mode,json.dumps(summary),flush=True)
        if args.repeat and all_results['hybrid']['summary']['ids_sha256']!=all_results['hybrid-repeat']['summary']['ids_sha256']:
            raise ValueError('Results differ across fresh processes')
    report={'archive':str(args.archive.resolve()),'sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            'synthetic_ocr':args.synthetic_ocr,'split_note':'Document-grouped for this recovery; public queries were used historically.',
            'modes':all_results}
    (args.output/('evaluation-ocr.json' if args.synthetic_ocr else 'evaluation.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
