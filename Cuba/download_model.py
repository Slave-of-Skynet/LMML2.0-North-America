"""Download only the pinned, standard MiniLM assets needed for offline building."""
import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
REVISION='b207367332321f8e44f96e224ef15bc607f4dbf0'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('destination',type=Path)
    args=ap.parse_args();args.destination.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors','onnx/model_quint8_avx2.onnx']:
        hf_hub_download(REPO,name,revision=REVISION,local_dir=args.destination)
    (args.destination/'source.json').write_text(json.dumps({'repo':REPO,'revision':REVISION}),encoding='utf-8')


if __name__=='__main__':main()
