from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
import yaml 
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', 
            default='/storage/feature/LibriTTS/sr_24000_mel_norm')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-train_index_file', default='train_samples_64.json')
    parser.add_argument('-logdir', default='log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('-store_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-load_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=5000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=0, type=int)

    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    solver = Solver(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)
