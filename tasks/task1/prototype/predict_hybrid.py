#!/usr/bin/env python3
"""Generate task1 predictions with a calibrated Baseline + Prototype hybrid."""
import argparse
import json
import os
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.task1.dataset import Dataset
from tasks.task1.model import Model
from .model import FramePrototypeModel


def collate(data, device=None):
    max_len=max(len(x[0]) for x in data); pad=lambda s:s+[0]*(max_len-len(s))
    return (torch.tensor([pad(x[0]) for x in data],dtype=torch.long,device=device),
            torch.tensor([pad(x[1]) for x in data],dtype=torch.long,device=device),
            [x[2] for x in data],[x[3] for x in data])


def collect(model, loader, device, kind, seen_mask=None, gamma=0.0):
    model.eval(); probs=[]; ids=[]
    with torch.no_grad():
        for input_ids, attention_mask, target, sentence_ids in tqdm(loader,desc=kind):
            if kind=='baseline':
                out=model(input_ids=input_ids,attention_mask=attention_mask,target=target,
                          labels=None,device=device,for_test=True)
            else:
                out=model(input_ids,attention_mask,target,labels=None,for_test=True)
            logits=out['logits']
            if kind=='prototype' and gamma!=0:
                logits=logits.clone(); logits[:,seen_mask]-=gamma
            probs.append(torch.softmax(logits,dim=-1).cpu()); ids.extend(sentence_ids)
    return torch.cat(probs),ids


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline_checkpoint',required=True)
    p.add_argument('--prototype_checkpoint',required=True)
    p.add_argument('--prototype_file',default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    p.add_argument('--baseline_weight',required=True,type=float)
    p.add_argument('--prototype_seen_gamma',default=0.0,type=float)
    p.add_argument('--batch_size',default=3,type=int)
    p.add_argument('--test_file',default='./dataset/cfn-test-B.json')
    p.add_argument('--train_file',default='./dataset/cfn-train.json')
    p.add_argument('--frame_info_file',default='./dataset/frame_info.json')
    p.add_argument('--config_file',default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file',default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--output_file',default='./tasks/task1/predictions/prototype-hybrid/A_task1_test.json')
    p.add_argument('--probability_file',default=None,
                   help='optional NPZ sidecar containing aligned 713-way frame probabilities')
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=BertTokenizer(vocab_file=args.vocab_file,do_lower_case=True)
    ds=Dataset(args.test_file,args.frame_info_file,tok,for_test=True)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,
                      collate_fn=partial(collate,device=device),drop_last=False)
    train=json.load(open(args.train_file,encoding='utf8')); seen={x['frame'] for x in train}
    seen_mask=torch.tensor([x in seen for x in ds.idx2label],dtype=torch.bool,device=device)

    cfg=BertConfig.from_json_file(args.config_file); cfg.num_labels=ds.num_labels
    cfg.model_variant='rope'; cfg.target_aware_layers='last'
    base=Model(cfg); base.load_state_dict(torch.load(args.baseline_checkpoint,map_location='cpu'),strict=False)
    base=base.to(device); bp,ids=collect(base,loader,device,'baseline'); del base
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    cfg2=BertConfig.from_json_file(args.config_file); cfg2.num_labels=ds.num_labels
    proto=FramePrototypeModel(cfg2,args.prototype_file)
    proto.load_state_dict(torch.load(args.prototype_checkpoint,map_location='cpu'),strict=False)
    proto=proto.to(device); pp,ids2=collect(proto,loader,device,'prototype',seen_mask,args.prototype_seen_gamma)
    if ids!=ids2: raise RuntimeError('Model outputs are not aligned.')
    scores=args.baseline_weight*bp+(1-args.baseline_weight)*pp
    pred=scores.argmax(-1).tolist()
    output=[[ids[i],ds.idx2label[pred[i]]] for i in range(len(ids))]
    os.makedirs(os.path.dirname(args.output_file) or '.',exist_ok=True)
    with open(args.output_file,'w',encoding='utf8',newline='\n') as f:
        json.dump(output,f,ensure_ascii=False,indent=1)
    if args.probability_file:
        if len(ds.all_data) != len(ids):
            raise RuntimeError('Task1 sidecar rows are not aligned with the test dataset.')
        raw_ids=np.asarray([int(x['sentence_id']) for x in ds.all_data],dtype=np.int64)
        if not np.array_equal(raw_ids,np.asarray(ids,dtype=np.int64)):
            raise RuntimeError('Task1 sentence IDs changed order while collecting probabilities.')
        target_starts=np.asarray([int(x['target'][-1]['start']) for x in ds.all_data],dtype=np.int32)
        target_ends=np.asarray([int(x['target'][-1]['end']) for x in ds.all_data],dtype=np.int32)
        os.makedirs(os.path.dirname(args.probability_file) or '.',exist_ok=True)
        np.savez_compressed(
            args.probability_file,
            frame_probs=scores.numpy().astype(np.float32),
            frame_names=np.asarray(ds.idx2label),
            sentence_ids=raw_ids,
            target_starts=target_starts,
            target_ends=target_ends,
            instance_indices=np.arange(len(ids),dtype=np.int32),
        )
        meta={
            'test_file':os.path.abspath(args.test_file),
            'rows':len(ids),
            'num_frames':ds.num_labels,
            'baseline_checkpoint':os.path.abspath(args.baseline_checkpoint),
            'prototype_checkpoint':os.path.abspath(args.prototype_checkpoint),
            'baseline_weight':args.baseline_weight,
            'prototype_weight':1-args.baseline_weight,
            'prototype_seen_gamma':args.prototype_seen_gamma,
            'probability_file':os.path.abspath(args.probability_file),
        }
        with open(args.probability_file+'.meta.json','w',encoding='utf8') as f:
            json.dump(meta,f,ensure_ascii=False,indent=2)
    print({'output_file':args.output_file,'rows':len(output),'baseline_weight':args.baseline_weight,
           'prototype_weight':1-args.baseline_weight,'prototype_seen_gamma':args.prototype_seen_gamma,
           'probability_file':args.probability_file})


if __name__=='__main__': main()

