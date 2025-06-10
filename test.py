import json
import logging

import torch
import pickle

from configs.settings import TotalConfigs
from eval import eval_fn
from models.trainer.evaluate import eval_language_metrics

logger = logging.getLogger(__name__)
def test_fn(cfgs: TotalConfigs, model, loader, device):
    print('##############n_vocab is {}##############'.format(cfgs.decoder.n_vocab))
    with open(cfgs.data.idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)
    with open(cfgs.data.vid2groundtruth_path, 'rb') as f:
        vid2groundtruth = pickle.load(f)
    scores = eval_fn(model=model, loader=loader, device=device, 
            idx2word=idx2word, save_on_disk=False, cfgs=cfgs,
            vid2groundtruth=vid2groundtruth)
    log_stats = {**{f'[test{k}': v for k, v in scores.items()},
                 }
    with open(cfgs.train.evaluate_dir, "a") as f:
        f.write(json.dumps(log_stats) + '\n')
    print('===================Testing is finished====================')

def test_fn_clip(cfgs: TotalConfigs, model, loader, device):
    print('##############n_vocab is {}##############'.format(model.caption_head.cap_config.vocab_size))
    checkpoint = {
        "epoch": -1,
        "cap_config": model.module.caption_head.cap_config if
        hasattr(model, 'module') else model.caption_head.cap_config
    }
    metrics = eval_language_metrics(checkpoint, loader, cfgs, model=model, device=device,
                                    eval_mode='test')

    logger.info('\t>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
                format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                       metrics['CIDEr'] * 100))
    epoch=-1
    log_stats = {**{f'[EPOCH{epoch + 1}]_test{k}': v for k, v in metrics.items()},
                 }
    with open(cfgs.train.evaluate_dir, "a") as f:
        f.write(json.dumps(log_stats) + '\n')
    print('===================Testing is finished====================')
