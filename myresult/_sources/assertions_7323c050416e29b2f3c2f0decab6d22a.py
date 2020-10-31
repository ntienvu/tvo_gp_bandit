import torch
import numpy as np
from src.ml_helpers import AverageMeter, get_grads, tensor, lognormexp, exponentiate_and_normalize, seed_all
from torch.distributions.multinomial import Multinomial

DUAL_OBJECTIVES = ['wake-wake', 'wake-sleep', 'tvo-sleep', 'tvo_reparam', 'tvo_reparam_iwae']
INTEGRATION_PARTITONS = ['left','right','trapz','single']
PCFGS = ['astronomers', 'brooks', 'minienglish', 'polynomial', 'quadratic', 'sids']
DISCRETE_LOSSES = ['reinforce','tvo','tvo_smoothed', 'vimco','wake-wake','wake-sleep', 'tvo-sleep']

# Assertions
def validate_hypers(args):
    assert args.schedule in [
        'log',
        'linear',
        'bq',
        'moments',
        'rand',
        'gp_bandit',
        'gp',
        'tvgp',
        #'beta_gradient_descent',
        #'beta_batch_gradient'
        ], f"schedule cannot be {args.schedule}"

    assert args.integration in INTEGRATION_PARTITONS, f"integration cannot be {args.integration}"

    assert args.integration_tvo_evidence in INTEGRATION_PARTITONS, f"integration_tvo_evidence cannot be {args.integration_tvo_evidence}"

    assert args.loss in [
        'reinforce',
        'elbo',
        'iwae',
        'tvo',
        'tvo_smoothed',
        'tvo_reparam',
        'tvo_reparam_iwae',
        'vimco',
        'wake-wake',
        'tvo-sleep',
        'wake-sleep'], f"loss cannot be {args.loss} "

    assert args.learning_task in [
        'continuous_vae',
        'discrete_vae',
        'bnn',
        'pcfg'
        ], f" learning_task cannot be {args.learning_task}"

    if args.learning_task != 'pcfg':
        assert args.dataset in [
                'tiny_mnist',
            'fashion_mnist',
            'mnist',
            'kuzushiji_mnist',
            'omniglot',
            'binarized_mnist',
            'binarized_omniglot'], f" dataset cannot be {args.dataset} "
    else:
        assert args.dataset in PCFGS, f"dataset must be one of {PCFGS}"
        assert args.loss in DISCRETE_LOSSES, f"loss can't be {args.loss} with {args.learning_task}"
        assert args.loss in ['tvo','tvo-sleep','wake-sleep', 'wake-wake'], f'{args.loss} not yet implemented for PCGS yet'

    if args.schedule != 'log':
       assert args.loss in ['elbo','tvo', 'tvo-sleep', 'tvo_reparam', 'tvo_smoothed', 'tvo_reparam_iwae'],  f"{args.loss} doesn't require a partition schedule scheme"
    if args.learning_task in ['discrete_vae']:
        assert args.dataset in ['binarized_mnist', 'binarized_omniglot'], \
            f" dataset cannot be {args.dataset} with {args.learning_task}"

        assert args.loss in DISCRETE_LOSSES, f"loss can't be {args.loss} with {args.learning_task}"

    if args.learning_task == 'bnn':
        assert args.dataset in ['fashion_mnist'], f" only fashion_mnist tested so far"
        assert args.loss not in DUAL_OBJECTIVES, f"BNN only has phi, can't use alternating objectives"

    if args.loss in DUAL_OBJECTIVES:
        assert not args.save_grads, 'Grad variance not able to handle duel objective methods yet'




    # Add an assertion everytime you catch yourself making a silly hyperparameter mistake so it doesn't happen again


def validate_dataset_path(args):
    learning_task = args.learning_task
    dataset = args.dataset

    if learning_task in ['discrete_vae', 'continuous_vae']:
        if dataset == 'fashion_mnist':
            data_path = args.data_dir + '/fashion_mnist.pkl'
        elif dataset == 'mnist':
            data_path = args.data_dir + '/mnist.pkl'
        elif dataset == 'tiny_mnist':
            data_path = args.data_dir + '/tiny_mnist.pkl'
        elif dataset == 'omniglot':
            data_path = args.data_dir + '/omniglot.pkl'
        elif dataset == 'kuzushiji_mnist':
            data_path = args.data_dir + '/kuzushiji_mnist.pkl'
        elif dataset == 'binarized_mnist':
            data_path = args.data_dir + '/binarized_mnist.pkl'
        elif dataset == 'binarized_omniglot':
            data_path = args.data_dir + '/binarized_omniglot.pkl'
    elif learning_task in ['bnn']:
        if dataset == 'fashion_mnist':
            data_path = args.data_dir + '/fmnist/'
    elif learning_task in ['pcfg']:
        data_path = args.data_dir + f'/pcfgs/{dataset}_pcfg.json'
    else:
        raise ValueError("Unknown learning task")

    return data_path
