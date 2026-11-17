#!/usr/bin/env python3
"""One-batch smoke test for Task2 A/B/C experiment switches."""
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model import Model
from .train import get_model_input


def build_model(config_file, checkpoint, num_labels, max_cls, valid_mask, device):
    config=BertConfig.from_json_file(config_file)
    config.num_labels=num_labels; config.max_cls=max_cls; config.task2_valid_loss_mask=valid_mask
    model=Model(config)
    msg=model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=False)
    return model.to(device),msg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config_file',default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file',default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--init_checkpoint',default='./chinese_bert_wwm_ext/pytorch_model.bin')
    p.add_argument('--batch_size',default=2,type=int)
    args=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=BertTokenizer(vocab_file=args.vocab_file,do_lower_case=True)
    ds=Dataset('./dataset/cfn-train.json','./dataset/frame_info.json',tok)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,
                      collate_fn=partial(get_model_input,device=device),drop_last=False)
    batch=next(iter(loader)); ids,mask,target,labels,_,_=batch
    legacy,msg1=build_model(args.config_file,args.init_checkpoint,1,ds.max_cls,False,device)
    valid,msg2=build_model(args.config_file,args.init_checkpoint,1,ds.max_cls,True,device)
    # The pretrained BERT checkpoint does not contain the randomly initialized
    # Task2 dense/LSTM parameters. Copy the full initialized state so A/B differ
    # only in the loss mask, not in random head initialization.
    valid.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval(); valid.eval()
    with torch.no_grad():
        out_a=legacy(input_ids=ids,attention_mask=mask,target=target,labels=labels,device=device)
        out_b=valid(input_ids=ids,attention_mask=mask,target=target,labels=labels,device=device)
    max_diff=(out_a['logits']-out_b['logits']).abs().max().item()
    print({'device':str(device),'logits_shape':list(out_a['logits'].shape),
           'legacy_loss':float(out_a['loss']),'valid_mask_loss':float(out_b['loss']),
           'max_logit_difference':max_diff,
           'legacy_missing':len(msg1.missing_keys),'valid_missing':len(msg2.missing_keys),
           'missing_key_examples':msg1.missing_keys[:12]})
    assert torch.isfinite(out_a['loss']) and torch.isfinite(out_b['loss'])
    assert max_diff < 1e-7, 'Loss-mask switch unexpectedly changed forward logits.'
    print('[OK] Task2 A/B/C one-batch smoke test passed.')


if __name__=='__main__': main()

