#!/usr/bin/python3

import argparse
import os


def construct_hyper_param():
    parser = argparse.ArgumentParser()
    # train settings
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size")
    parser.add_argument("--update_freq", default=1, type=int,
                        help="update_freq")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help='model learning rate.')

    # experiment bookkeeping
    parser.add_argument('--experiment_name', default='baseline', type=str,
                        help='experiment name; metrics/checkpoints go to <name>-result')
    parser.add_argument('--result_dir', default=None, type=str,
                        help='optional explicit result directory')
    parser.add_argument('--log_interval', default=100, type=int,
                        help='print running loss every N steps')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for reproducible comparisons')
    parser.add_argument('--test_file', default='./dataset/cfn-test-B.json', type=str,
                        help='prediction input file; default is test-B')
    parser.add_argument('--prediction_dir', default=None, type=str,
                        help='directory for generated prediction JSON files')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='optional explicit checkpoint path for prediction')
    parser.add_argument('--model_variant', default='rope', choices=['rope', 'target_aware'],
                        help='task1 architecture; rope keeps the baseline, target_aware is the v1 experiment')
    parser.add_argument('--target_aware_layers', default='last', choices=['last', 'last4'],
                        help='hidden-layer source for target-aware task1')
    parser.add_argument('--task1_target_markers', action='store_true',
                        help='insert [unused1]/[unused2] around the task1 target span')
    parser.add_argument('--task2_threshold', default=0.0, type=float,
                        help='Task2 span decoding threshold')
    parser.add_argument('--task2_safe_decode', action='store_true',
                        help='restrict Task2 predictions to valid character spans outside target')
    parser.add_argument('--task2_valid_loss_mask', action='store_true',
                        help='compute Task2 loss only on valid character spans outside target')
    parser.add_argument('--task2_disable_fgm', action='store_true',
                        help='disable FGM adversarial training for Task2 ablation')
    parser.add_argument('--task2_disable_parameter_noise', action='store_true',
                        help='disable legacy one-time parameter noise (recommended for warm-start experiments)')
    parser.add_argument('--task2_model_variant', default='baseline',
                        choices=['baseline', 'target_conditioned', 'content_aware',
                                 'frame_prototype_conditioned', 'boundary_auxiliary'],
                        help='Task2 architecture variant')
    parser.add_argument('--task2_content_dim', default=32, type=int,
                        help='low-dimensional span-content projection size for Task2 content-aware model')
    parser.add_argument('--task2_content_scale_init', default=0.1, type=float,
                        help='initial additive content-score scale for Task2 content-aware model')
    parser.add_argument('--task2_frame_prototype_file',
                        default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt',
                        help='fixed frame-definition prototypes for oracle-frame Task2 experiments')
    parser.add_argument('--task2_frame_query_dim', default=64, type=int,
                        help='frame prototype query projection dimension')
    parser.add_argument('--task2_frame_scale_init', default=0.1, type=float,
                        help='initial frame-prototype residual scale')
    parser.add_argument('--task2_boundary_loss_weight', default=0.1, type=float,
                        help='weight of H1 start/end boundary auxiliary loss')
    parser.add_argument('--task2_boundary_residual_beta', default=0.0, type=float,
                        help=('optional H1-R inference residual: span(i,j) + beta * '
                              '(start(i) + end(j)); zero preserves H1 span-only decoding'))
    parser.add_argument('--task2_aux_head_lr', default=None, type=float,
                        help='optional LR for newly added Task2 auxiliary heads')
    parser.add_argument('--task3_frame_constrained', action='store_true',
                        help='restrict Task3 FE predictions to roles licensed by the Task1 frame')
    parser.add_argument('--task1_prediction_file', default=None, type=str,
                        help='Task1 official prediction JSON used by frame-constrained Task3 decoding')
    parser.add_argument('--task2_prediction_file', default=None, type=str,
                        help='Task2 span prediction JSON consumed by Task3 (overrides prediction_dir lookup)')

    # bert settings
    parser.add_argument("--config_file",
                        default='./chinese_bert_wwm_ext/config.json',
                        type=str,
                        help="bert_config_file")
    parser.add_argument("--vocab_file",
                        default='./chinese_bert_wwm_ext/vocab.txt',
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint",
                        default='./chinese_bert_wwm_ext/pytorch_model.bin',
                        type=str,
                        help="bert_init_checkpoint")

    parser.add_argument("--with_adv_train",
                        action='store_true',
                        help="with_adv_train")

    parser.add_argument("--data_dir",
                        default='./data', type=str,
                        help="save_dir")

    parser.add_argument("--save_dir",
                        default='./models', type=str,
                        help="save_dir")

    args = parser.parse_args()

    return args



args = construct_hyper_param()







