#!/usr/bin/env python3
"""Validate Task2 span coordinates, marker insertion, and data statistics."""
import argparse
import json
from collections import Counter

from transformers import BertTokenizer

from .dataset import Dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vocab_file',default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--frame_info_file',default='./dataset/frame_info.json')
    p.add_argument('--max_samples',default=0,type=int)
    args=p.parse_args()
    tok=BertTokenizer(vocab_file=args.vocab_file,do_lower_case=True)
    for path in ['./dataset/cfn-train.json','./dataset/cfn-dev.json']:
        raw=json.load(open(path,encoding='utf8'))
        ds=Dataset(path,args.frame_info_file,tok)
        n=len(ds) if args.max_samples<=0 else min(len(ds),args.max_samples)
        errors=[]; ignored=0; total_labels=0; nested=0; lengths=[]
        for i in range(n):
            input_ids,mask,target,labels,sid,_=ds[i]; row=raw[i]; s,e=target
            if len(input_ids)!=len(row['text'])+4: errors.append((i,'length',len(input_ids),len(row['text'])+4))
            if input_ids[s]!=1 or input_ids[e+2]!=2: errors.append((i,'markers',s,e,input_ids[s],input_ids[e+2]))
            if len(mask)!=len(input_ids) or not all(x==1 for x in mask): errors.append((i,'mask'))
            expected=[]
            for span in row['cfn_spans']:
                lengths.append(span['end']-span['start']+1)
                if span['end']<row['target'][-1]['start']:
                    expected.append([span['start']+1,span['end']+1])
                elif span['start']>row['target'][-1]['end']:
                    expected.append([span['start']+3,span['end']+3])
                else: ignored+=1
            if labels!=expected: errors.append((i,'labels',labels,expected))
            total_labels+=len(labels)
            for a_i,a in enumerate(labels):
                for b in labels[a_i+1:]:
                    if (a[0]<=b[0] and a[1]>=b[1]) or (b[0]<=a[0] and b[1]>=a[1]): nested+=1
        print({'split':path,'checked':n,'errors':len(errors),'usable_labels':total_labels,
               'ignored_target_overlap':ignored,'nested_pairs':nested,
               'max_sequence_length':max(len(ds[i][0]) for i in range(n)),
               'mean_span_length':sum(lengths)/max(len(lengths),1)})
        if errors:
            print(errors[:20]); raise AssertionError(f'{path}: Task2 validation failed')
    print('[OK] Task2 data/coordinate checks passed.')


if __name__=='__main__': main()

