import torch
import numpy as np
from src.ml_helpers import AverageMeter, get_grads, tensor, lognormexp, exponentiate_and_normalize, seed_all
from torch.distributions.multinomial import Multinomial
import src.ml_helpers as mlh
#from src.GPv import GaussianProcessDerivative

def range_except(end, i):
    """Outputs an increasing list from 0 to (end - 1) except i.
    Args:
        end: int
        i: int

    Returns: list of length (end - 1)
    """

    result = list(set(range(end)))
    return result[:i] + result[(i + 1):]


def get_partition(args):
    """Create a non-decreasing sequence of values between zero and one.
    See https://en.wikipedia.org/wiki/Partition_of_an_interval.

    Args:
        args.num_partitions: length of sequence minus one
        args.schedule: \'linear\' or \'log\'
        args.log_beta_min: log (base ten) of beta_min. only used if partition_type
            is log. default -10 (i.e. beta_min = 1e-10).
        args.device: torch.device object (cpu by default)

    Returns: tensor of shape [num_partitions + 1]
    """
    if args.K == 1:
        partition = tensor((0., 1), args)
    else:
        if args.schedule == 'linear':
            partition = torch.linspace(0, 1, steps=args.K + 1,
                                       device=args.device)
        elif args.schedule == 'log':
            partition = torch.zeros(args.K + 1, device=args.device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(args.log_beta_min, 0, steps=args.K, device=args.device,
                                           dtype=torch.float)
        else:
            # DEFAULT IS TO START WITH LOG
            partition = torch.zeros(args.K + 1, device=args.device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(args.log_beta_min, 0, steps=args.K, device=args.device,
                                           dtype=torch.float)
    return partition


def _get_multiplier(partition, integration):
    """ Outputs partition multipers depending on integration rule
     Args:
         partition = partition of interval [0,1]
         integration : left, right, trapz, single (i.e. 1 * partition[*, 1])

        (helper function to accomodate per_sample calculations)

     Returns: tensor with size = partition.shape
     """
    if len(partition.shape) == 1:
        multiplier = torch.zeros_like(partition)
        if integration == 'trapz':
            multiplier[0] = 0.5 * (partition[1] - partition[0])
            multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
            multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
        elif integration == 'left':
            multiplier[:-1] = partition[1:] - partition[:-1]
        elif integration == 'right':
            multiplier[1:] = partition[1:] - partition[:-1]

    else:
        multiplier = torch.zeros_like(partition)
        if integration == 'trapz':
            multiplier[..., 0] = 0.5 * (partition[1] - partition[0])
            multiplier[..., 1:-1] = 0.5 * (partition[2:] - partition[0:-2])
            multiplier[..., -1] = 0.5 * (partition[-1] - partition[-2])
        elif integration == 'left':
            multiplier[..., :-1] = partition[..., 1:] - partition[..., :-1]
        elif integration == 'right':
            multiplier[..., 1:] = partition[..., 1:] - partition[..., :-1]
        elif integration == 'single':
            multiplier = torch.ones_like(partition)
            if multiplier.shape[-1] == 3:
                multiplier[..., 0] = 0
                multiplier[..., -1] = 0

    return multiplier


def compute_tvo_loss(log_weight, log_p, log_q, args):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data

    """
    partition = args.partition
    num_particles = args.S
    integration = args.integration

    log_weight = log_weight.unsqueeze(-1)

    heated_log_weight = log_weight * partition
    heated_normalized_weight = exponentiate_and_normalize(
        heated_log_weight, dim=1)

    thermo_logp = partition * log_p.unsqueeze(-1) + \
        (1 - partition) * log_q.unsqueeze(-1)

    wf = heated_normalized_weight * log_weight
    w_detached = heated_normalized_weight.detach()
    wf_detached = wf.detach()

    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    cov = correction * torch.sum(
        w_detached * (log_weight - torch.sum(wf, dim=1, keepdim=True)).detach() *
        (thermo_logp - torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
        dim=1)

    multiplier = _get_multiplier(partition, integration)

    loss = -torch.mean(torch.sum(
        multiplier * (cov + torch.sum(
            w_detached * log_weight, dim=1)),
        dim=1))

    return loss



def compute_tvo_log_evidence(log_weight, args):

    # this is special partition for tvo_log_evidence

    partition = mlh.tensor(10**args.partition_tvo_evidence, args)
    # this is the integration used for computing tvo_log_evidence only
    # should be trapz
    integration = args.integration_tvo_evidence

    log_weight = log_weight.unsqueeze(-1)

    heated_log_weight = log_weight * partition
    heated_normalized_weight = exponentiate_and_normalize(heated_log_weight, dim=1)
    Epi_beta = torch.sum(heated_normalized_weight * log_weight, dim=1)

    multiplier = _get_multiplier(partition, integration)

    tvo_logpx = torch.sum(multiplier * Epi_beta, dim=1)

    return tvo_logpx

def compute_tvo_iwae_log_evidence(log_weight, args):
    # see notes @ BQD_TVO/notes/tvo_iwae.pdf
    partition = args.partition
    S = log_weight.shape[1]

    log_weight = log_weight.unsqueeze(-1)
    heated_log_weight = log_weight * partition

    log_evidence = torch.logsumexp(heated_log_weight, dim=1) - np.log(S)
    tvo_iwae_log_evidence = torch.sum(log_evidence * 1/partition.sum(), dim=1)

    return tvo_iwae_log_evidence


def calculate_grad_variance(model, args):
    grad_var = AverageMeter()
    batch = next(iter(args.train_data_loader))
    for _ in range(10):
        model.zero_grad()
        loss, logpx, test_elbo = model.forward(batch)
        loss.backward()
        model_grads = get_grads(model)
        grad_var.step(model_grads)

    grad_std = grad_var.variance.sqrt().mean()
    return grad_std


def compute_wake_phi_loss(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def compute_wake_theta_loss(log_weight):
    """Args:
    log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(torch.logsumexp(
        log_weight, dim=1) - np.log(num_particles))
    return -elbo


def compute_vimco_loss(log_weight, log_q):
    num_particles = log_q.shape[1]

    # shape [batch_size, num_particles]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=1, keepdim=True) - log_weight) \
        / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(
        log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - np.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(torch.sum(
        (log_evidence.unsqueeze(-1) - control_variate).detach() * log_q, dim=1
    ))

    return loss


def calc_exp(log_weight, args, all_sample_mean=True, snis=None):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : True averages over batch

    TO DO : replace for cleaner integration into code (pulled directly from Rob's)
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight
    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = snis * log_weight
    exp = torch.sum(exp, dim=1)
    return torch.mean(exp, dim=0) if all_sample_mean else exp

def calc_var_given_betas(log_weight, args, all_sample_mean=True, betas = None):
    """
    Args:
        log_weight : [batch, samples, *]
        args : args object
                *** Note: only args.partition is used (this can be replaced by betas arg)
        all_sample_mean : returns mean over samples if True
        betas : β points at which to evaluate variance (shape: [K β points] or  [batch or 1, 1, K β points])
    Returns:
        Variance across importance samples at each beta (2nd derivative of log Z_β wrt β)
    """

    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight


    if betas is None:
        partition = args.partition
    else:
        partition = mlh.tensor(betas, args)

    beta_iw = log_weight * partition
    snis = exponentiate_and_normalize(
        beta_iw, dim=1)
    snis_detach = snis.detach()

    # expected log weights under π_β
    exp_pi_beta_log_weight  = torch.sum(snis_detach*log_weight, dim=1, keepdim=True)

    variance = torch.sum(snis_detach*torch.pow(log_weight - exp_pi_beta_log_weight, 2), dim =1 )

    # variance is [batch, # β].  set all_sample_mean = True to average over batch dimension
    return torch.mean(variance, dim=0) if all_sample_mean else variance

def calc_var(log_weight,  args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Variance across importance samples at each beta (2nd derivative of logZβ)
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp_ = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)

    to_return = exp2 - torch.pow(exp_, 2)

    # VM: May have to switch to_return to E[(X-EX)(X-EX)] form, had numerical issues in the past
    assert not torch.isnan(to_return).any(), "Nan in calc_var() - switch to E[(X-EX)(X-EX)] form for numerical stability"


    return torch.mean(to_return, dim=0) if all_sample_mean else to_return


#
def calc_third(log_weight, args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Third derivative of logZβ at each beta
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)
    var = exp2 - torch.pow(exp, 2)
    exp3 = torch.sum(snis * torch.pow(log_weight, 3), dim=1)

    to_return = exp3 - torch.pow(exp, 3) - 3*exp*var
    return torch.mean(to_return, dim=0) if all_sample_mean else to_return


def calc_fourth(log_weight, args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Fourth derivative of logZβ at each beta
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)
    exp3 = torch.sum(snis * torch.pow(log_weight, 3), dim=1)
    exp4 = torch.sum(snis * torch.pow(log_weight, 4), dim=1)

    to_return = exp4 - 6*torch.pow(exp, 4) + 12*exp2 * \
        torch.pow(exp, 2) - 3*torch.pow(exp2, 2) - 4*exp*exp3
    return torch.mean(to_return, dim=0) if all_sample_mean else to_return


def get_curvature_loss_and_grad(beta_model, model, args):
    # Refers to writeup here: tvo_icml/notes/curvature_optimization.pdf
    log_weight = get_total_log_weight(model, args, args.valid_S).unsqueeze(-1)

    log_weight = log_weight
    partition = beta_model.weight.data.squeeze(1)

    snis = exponentiate_and_normalize(log_weight * partition, dim=1)

    # Calling them first, second, third to match notation in writeup
    first  = calc_var(log_weight, args, snis=snis, all_sample_mean=True)
    second = calc_third(log_weight, args, snis=snis, all_sample_mean=True)
    third  = calc_fourth(log_weight, args, snis=snis, all_sample_mean=True)

    unsigned_curvature = second.abs() / torch.pow(1 + torch.pow(first, 2), 3 / 2)
    log_curvature = torch.log(second.abs()) - (1.5) * torch.log(1 + torch.pow(first, 2))
    grad = (third / second) - ((3 * first * second) / (1 - torch.pow(first, 2)))

    grad.unsqueeze_(1)

    return -log_curvature, grad, unsigned_curvature


class ChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist, get_next_dist):
        self.chain_dist = chain_dist
        self.get_next_dist = get_next_dist

    def sample(self, sample_shape=torch.Size()):
        sample_chain = self.chain_dist.sample(sample_shape=sample_shape)
        sample_next = self.get_next_dist(sample_chain[-1]).sample(
            sample_shape=())
        return sample_chain + (sample_next,)

    def rsample(self, sample_shape=torch.Size()):
        sample_chain = self.chain_dist.rsample(sample_shape=sample_shape)
        sample_next = self.get_next_dist(sample_chain[-1]).rsample(
            sample_shape=())
        return sample_chain + (sample_next,)

    def log_prob(self, value):
        log_prob_chain = self.chain_dist.log_prob(value[:-1])
        log_prob_next = self.get_next_dist(value[-2]).log_prob(value[-1])
        return log_prob_chain + log_prob_next


class ChainDistributionFromSingle(torch.distributions.Distribution):
    def __init__(self, single_dist):
        self.single_dist = single_dist

    def sample(self, sample_shape=torch.Size()):
        return (self.single_dist.sample(sample_shape=sample_shape),)

    def rsample(self, sample_shape=torch.Size()):
        return (self.single_dist.rsample(sample_shape=sample_shape),)

    def log_prob(self, value):
        return self.single_dist.log_prob(value[0])


class ReversedChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist):
        self.chain_dist = chain_dist

    def sample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.sample(
            sample_shape=sample_shape)))

    def rsample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.rsample(
            sample_shape=sample_shape)))

    def log_prob(self, value):
        return self.chain_dist.log_prob(tuple(reversed(value)))


def get_total_log_weight(model, args, S):
    with torch.no_grad():
        log_weight = []
        for obs in args.train_data_loader:
            model.set_internals(obs, S)
            elbo = model.elbo()
            log_weight.append(elbo)

        log_weight = torch.cat(log_weight)

    return log_weight

def get_total_log_weight_joint_guide(model, args, S):
    with torch.no_grad():
        log_weight = []
        log_joint = []
        log_guide = []

        for obs in args.train_data_loader:
            model.set_internals(obs, S)
            elbo = model.elbo()
            pp = model.log_joint()
            qq = model.log_guide()
            log_weight.append(elbo)
            log_joint.append(pp)
            log_guide.append(qq)

        log_weight = torch.cat(log_weight)
        log_joint = torch.cat(log_joint)
        log_guide = torch.cat(log_guide)

    return log_weight,log_joint,log_guide

def compute_tvo_reparam_loss(log_weight, log_p, log_q, args, return_full = False, amci = False):
    num_particles = args.S


    log_weight = log_weight.unsqueeze(-1) if len(log_weight.shape)<3 else log_weight
    if args.iw_resample:
        log_weight, log_p, log_q, heated_normalized_weight, zb = iw_resampler(log_weight, log_p, log_q, args)

    snis_logw, thermo_logp, snis_detach = get_tvo_components(log_weight, log_p, log_q, args, \
                                            heated_normalized_weight= heated_normalized_weight if args.iw_resample else None)


    beta_one_minus = args.partition*(1-args.partition)
    one_minus_two_beta =  torch.ones_like(args.partition)-2*args.partition

    tvo_reparam = beta_one_minus.squeeze()*torch.sum(snis_detach* \
        (log_weight.detach() - torch.sum(snis_logw, dim=1,keepdim=True).detach()) \
            * (log_weight - torch.sum(snis_detach*log_weight, dim=1,keepdim=True)), dim=1) \
                + one_minus_two_beta.squeeze() * torch.sum(snis_detach*log_weight, dim=1)

    if return_full:
        return -tvo_reparam
    else:
        multiplier = _get_multiplier(args.partition, args.integration).squeeze()
        return -torch.mean(torch.sum(multiplier*tvo_reparam, dim=1), dim=0)

def get_tvo_components(log_weight, log_p, log_q, args, heated_normalized_weight = None):

    partition = args.partition
    num_particles = args.S
    integration = args.integration

    # feed heated_normalized_weight when doing importance resampling (due to uniform expectations at selected indices)
    if heated_normalized_weight is None:

        heated_log_weight = log_weight * partition
        heated_normalized_weight = exponentiate_and_normalize(
            heated_log_weight, dim=1)

    log_p = log_p.unsqueeze(-1) if len(log_p.shape)<3 else log_p
    log_q = log_q.unsqueeze(-1) if len(log_q.shape)<3 else log_q
    thermo_logp = partition * log_p + \
            (1 - partition) * log_q

    snis_logw = heated_normalized_weight * log_weight
    snis_detach = heated_normalized_weight.detach()

    return snis_logw, thermo_logp, snis_detach
