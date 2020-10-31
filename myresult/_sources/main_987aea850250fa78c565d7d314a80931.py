import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm

import src.ml_helpers as mlh
from src import assertions, util
from src.assertions import DUAL_OBJECTIVES
from src.bayes_quad import format_input#, get_integrand_function
from src.data_handler import get_data
from src.models import updates
from src.models.model_handler import get_model

#from src.data import get_data_loader

warnings.filterwarnings("ignore")

ex = Experiment()

torch.set_printoptions(sci_mode=False)


@ex.config
def my_config():
    """
    This specifies all the parameters for the experiment.
    Only native python objects can appear here (lists, string, dicts, are okay,
    numpy arrays and tensors are not). Everything defined here becomes
    a hyperparameter in the args object, as well as a column in omniboard.
    More complex objects are defined and manuipulated in the init() function
    and attached to the args object.
    The ProbModelBaseClass object is stateful and contains self.args,
    so hyperparameters are accessable to the model via self.args.hyper_param
    """

    # learning task
    learning_task = 'continuous_vae'
    #learning_task = 'discrete_vae'
    artifact_dir = './artifacts'
    data_dir = './data'

    # Model
    loss = 'tvo'
    hidden_dim = 100  # Hidden dimension of middle NN layers in vae
    latent_dim = 25  # Dimension of latent variable z
    integration = 'left'
    integration_tvo_evidence = 'trapz'
    # this is used to estimate get_tvo_log_evidence only
    partition_tvo_evidence = np.linspace(-9, 0, 50)

    cuda = True
    num_stochastic_layers = 1
    num_deterministic_layers = 2
    learn_prior = False
    activation = None  # override Continuous VAE layers
    iw_resample = False  # whether to importance resample TVO proposals (WIP)

    # to terminate a chosen beta for another one if the logpx drops more than this threshold
    drip_threshold = -0.05
    # if it is terminated, this indicates how many epochs have been run from the last bandit
    len_terminated_epoch = 0

    # Hyper
    K = 5
    S = 10
    lr = 0.001
    log_beta_min = -1.602  # -1.09
    bandit_beta_min = 0.05  # -1.09
    bandit_beta_max = 0.95  # -1.09

    # Scheduling
    schedule = 'gp_bandit'
    burn_in = 20  # number of epochs to wait before scheduling begins, useful to set low for debugging
    schedule_update_frequency = 6  # if 0, initalize once and never update
    per_sample = False  # Update schedule for each sample
    per_batch = False

    # Recording
    record = False
    record_partition = None #True  # unused.  possibility to std-ize partitions for evaluation
    verbose = False
    dataset = 'mnist'
    #dataset = 'omniglot'

    phi_tag = 'encoder'
    theta_tag = 'decoder'

    # Training
    seed = 1
    epochs = 5000
    batch_size = 1000  # 1000
    valid_S = 100
    test_S = 5000
    test_batch_size = 1

    increment_update_frequency=10


    optimizer = "adam"
    checkpoint_frequency = int(epochs / 5)
    checkpoint = False
    checkpoint = checkpoint if checkpoint_frequency > 0 else False

    test_frequency = 200  # 20
    test_during_training = True
    test_during_training = test_during_training if test_frequency > 0 else False
    train_only = False
    save_grads = False

    # store all betas and logpx at all epochs
    betas_all = np.empty((0, K+1), float)
    logtvopx_all = []
    truncation_threshold = 30*K
    X_ori = np.empty((0, K+1), float)
    Y_ori = []
    average_y = []


    # beta gradient descent step size
    beta_step_size = 0.01
    max_beta_step = 0.025
    adaptive_beta_step = False

    # following args all set internaly
    init_expectation = None
    expectation_diffs = 0 # mlh.AccumulatedDiff()

    if learning_task == 'discrete_vae':
        dataset = 'binarized_mnist'
        # dataset = 'binarized_omniglot'

        # To match paper (see app. I)
        num_stochastic_layers = 3
        num_deterministic_layers = 0
        increment_update_frequency=10


    if learning_task == 'bnn':
        dataset = 'fashion_mnist'

        bnn_mini_batch_elbo = True

        batch_size = 100 # To match tutorial (see: https://www.nitarshan.com/bayes-by-backprop/)
        test_batch_size = 5

        # This can still be overwritten via the command line
        S = 10
        test_S = 10
        valid_S = 10

    if learning_task == 'pcfg':
        dataset = 'astronomers'
        ## to match rrws code
        batch_size = 2
        schedule = 'log'
        S = 20
        train_only = True # testing happens in training loop
        cuda = False
        epochs = 2000

        phi_tag = 'inference_network'
        theta_tag = 'generative_model'


def init(config, _run):
    args = SimpleNamespace(**config)
    assertions.validate_hypers(args)
    mlh.seed_all(args.seed)

    args.data_path = assertions.validate_dataset_path(args)

    if args.activation is not None:
        if 'relu' in args.activation:
            args.activation = torch.nn.ReLU()
        elif 'elu' in args.activation:
            args.activation = torch.nn.ELU()
        else:
            args.activation = torch.nn.ReLU()

    args._run = _run

    Path(args.artifact_dir).mkdir(exist_ok=True)

    args.loss_name = args.loss

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    args.partition_scheduler = updates.get_partition_scheduler(args)
    args.partition = util.get_partition(args)

    args.data_path = Path(args.data_path)
    return args


@ex.capture
def log_scalar(_run=None, **kwargs):
    assert "step" in kwargs, 'Step must be included in kwargs'
    step = kwargs.pop('step')

    for k, v in kwargs.items():
        _run.log_scalar(k, float(v), step)

    loss_string = " ".join(("{}: {:.4f}".format(*i) for i in kwargs.items()))
    print(f"Epoch: {step} - {loss_string}")


@ex.capture
def save_checkpoint(model, epoch, train_elbo, train_logpx, opt, args, _run=None, _config=None):
    path = Path(args.artifact_dir) / 'model_epoch_{:04}.pt'.format(epoch)

    print("Saving checkpoint: {}".format(path))

    if args.loss in DUAL_OBJECTIVES:
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_phi': opt[0].state_dict(),
                    'optimizer_theta': opt[1].state_dict(),
                    'train_elbo': train_elbo,
                    'train_logpx': train_logpx,
                    'config': dict(_config)}, path)
    else:
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': opt[0].state_dict(),
                    'train_elbo': train_elbo,
                    'train_logpx': train_logpx,
                    'config': dict(_config)}, path)

    _run.add_artifact(path)


def train(args):
    # read data
    train_data_loader, test_data_loader = get_data(args)

    # attach data to args
    args.train_data_loader = train_data_loader
    args.test_data_loader = test_data_loader

    # Make models
    model = get_model(train_data_loader, args)

    # Make optimizer
    if args.loss in DUAL_OBJECTIVES:
        optimizer_phi = torch.optim.Adam(
            (params for name, params in model.named_parameters() if args.phi_tag in name), lr=args.lr)
        optimizer_theta = torch.optim.Adam(
            (params for name, params in model.named_parameters() if args.theta_tag in name), lr=args.lr)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #for epoch in range(args.epochs):
    for epoch in tqdm(range(args.epochs)):
        if mlh.is_schedule_update_time(epoch, args):
                args.partition = args.partition_scheduler(model, args)
                if len(args.Y_ori)%args.increment_update_frequency==0 and len(args.Y_ori)>1:
                    args.schedule_update_frequency=args.schedule_update_frequency+1
                    print("args.schedule_update_frequency=",args.schedule_update_frequency)


        if args.loss in DUAL_OBJECTIVES:
            train_logpx, train_elbo, train_tvo_log_evidence = model.train_epoch_dual_objectives(
                train_data_loader, optimizer_phi, optimizer_theta, epoch=epoch)
        else:
            # addl recording within model.base
            train_logpx, train_elbo, train_tvo_log_evidence = model.train_epoch_single_objective(
                train_data_loader, optimizer, epoch=epoch)

        log_scalar(train_elbo=train_elbo, train_logpx=train_logpx,
                   train_tvo_log_evidence=train_tvo_log_evidence, step=epoch)

        # store the information
        args.betas_all = np.vstack((args.betas_all, np.reshape(
            format_input(args.partition), (1, args.K+1))))
        args.logtvopx_all = np.append(
            args.logtvopx_all, train_tvo_log_evidence)


        if mlh.is_gradient_time(epoch, args):
            # Save grads
            grad_variance = util.calculate_grad_variance(model, args)
            log_scalar(grad_variance=grad_variance, step=epoch)

        if mlh.is_test_time(epoch, args):
            test_logpx, test_kl = model.evaluate_model_and_inference_network(test_data_loader, epoch=epoch)
            log_scalar(test_logpx=test_logpx, test_kl=test_kl, step=epoch)

        if mlh.is_checkpoint_time(epoch, args):
            opt = [optimizer_phi, optimizer_theta] if args.loss in DUAL_OBJECTIVES else [optimizer]
            save_checkpoint(model, epoch, train_elbo, train_logpx, opt, args)


        # ------ end of training loop ---------
    opt = [optimizer_phi, optimizer_theta] if args.loss in DUAL_OBJECTIVES else [optimizer]
    save_checkpoint(model, args.epochs, train_elbo, train_logpx, opt, args)

    if args.train_only:
        test_logpx, test_kl = 0, 0

    results = {
        "test_logpx": test_logpx,
        "test_kl": test_kl,
        "train_logpx": train_logpx,
        "train_elbo": train_elbo,
        "train_tvo_px": train_tvo_log_evidence,
        "average_y": args.average_y,  # average tvo_logpx within this bandit iteration
        "X": args.X_ori,  # this is betas
        # this is utility score y=f(betas)= ave_y[-1] - ave_y[-2]
        "Y": args.Y_ori
    }

    return results, model


@ex.automain
def experiment(_config, _run):
    '''
    Amended to return
    '''

    args = init(_config, _run)
    result, model = train(args)

    if args.record:
        model.record_artifacts(_run)

    return result
