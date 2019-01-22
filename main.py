from argparse import ArgumentParser
import torch
from solver import Solver
import yaml 
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', 
            default='/storage/feature/voice_conversion/vctk_waveform/librosa/split_10_0.1')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-val_set', default='in_test')
    parser.add_argument('-train_index_file', default='train_samples.json')
    parser.add_argument('-val_index_file', default='in_test_samples.json')
    parser.add_argument('-logdir', default='log/')
    parser.add_argument('-load_model', action='store_true')
    parser.add_argument('-load_opt', action='store_true')
    parser.add_argument('-load_dis', action='store_true')
    parser.add_argument('-store_model_path', default='/storage/model/adaptive_vc/')
    parser.add_argument('-load_model_path', default='/storage/model/adaptive_vc/model')

    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    solver = Solver(config=config, args=args)
