import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from train import run


def main():
    parser = argparse.ArgumentParser(description='Fast weights Training')

    parser.add_argument('--exp', default='all', help='run a single experiment or all experiments')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    exp_files = sorted(glob.glob('exps/*/config.yml'))
    exp_already_run = [str(Path(f).parent / 'config.yml') for f in glob.glob('exps/*/*_testinglog.npy')]
    exp_to_run = list(set(exp_files) - set(exp_already_run))
    if opts.exp == 'all':
        for exp in exp_to_run:
            with open(exp, 'r') as f:
                args = yaml.safe_load(f)
            run(args, random_seed=opts.seed)
    elif opts.exp in exp_files:
        if opts.exp in exp_to_run:
            with open(opts.exp, 'r') as f:
                args = yaml.safe_load(f)
            run(args, random_seed=opts.seed)
        else:
            raise Exception('This is a done experiment')
    else:
        raise Exception('Unknown experiment')


if __name__ == "__main__":
    main()
