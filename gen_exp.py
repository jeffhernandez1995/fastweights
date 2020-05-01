import os

import yaml
import itertools
import numpy as np


def base_args():
    base_ = {
        'name': None,
        'dir': None,
        'log_dir': 'logs/',
        'config': {
            'batch_size': 128,
            'device': 'cuda',
            'epochs': 200,
            'workers': 16,
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'scheduler': 'none',
            'lr_step_size': 50,
            'lr_steps': [100, 150],
            'lr_gamma': 0.1,
            'reduce_type': 'min',
            'max_norm': 0,
            'output_dir': 'models/',
            'patience': 200,
            'resume': 0,
            'model': {
                'input_size': 26 + 10 + 1,
                'hidden_size': 128,
                'out_size': 26 + 10 + 1,
                'fast_lr': 0.5,
                'decay_lr': 0.95,
                'control': True,
                'layer_norm': False,
                'extra_depth': False,
                'device': 'cuda'
            }
        }
    }
    return base_


def main():
    experiments = list(itertools.product(
        [False],  # Control
        [False, True],  # layer_norm
        [False, True],  # extra_depth
        [False, True]   # hidden size
    ))
    experiments.append((True, False, False, False))
    BASE_DIR = 'exps/'

    for exp in sorted(experiments, reverse=True):
        args = base_args()
        args['config']['model']['control'] = exp[0]
        args['config']['model']['layer_norm'] = exp[1]
        args['config']['model']['extra_depth'] = exp[2]
        if exp[3]:
            args['config']['model']['hidden_size'] = 128
        else:
            args['config']['model']['hidden_size'] = 64
        exp_int = np.array(exp).astype(int)
        args['name'] = 'exp_' + '_'.join([str(f) for f in exp_int])
        args['dir'] = os.path.join(BASE_DIR, f"{args['name']}/")

        if not os.path.exists(os.path.join(args['log_dir'], args['name'])):
            os.mkdir(os.path.join(args['log_dir'], args['name']))

        if not os.path.exists(args['dir']):
            os.mkdir(args['dir'])
            os.mkdir(os.path.join(args['dir'], args['config']['output_dir']))
            with open(f"{args['dir']}/config.yml", 'w') as yaml_file:
                yaml.dump(args, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    main()
