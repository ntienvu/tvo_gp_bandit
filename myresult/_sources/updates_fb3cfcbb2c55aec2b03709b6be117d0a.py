import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
import copy

#import torch.nn as nn
from functools import partial
import numpy as np
from src.util import compute_tvo_loss, compute_wake_theta_loss, compute_wake_phi_loss, get_curvature_loss_and_grad
from src.util import compute_vimco_loss, exponentiate_and_normalize
from src.util import calc_exp, calc_var_given_betas, get_total_log_weight, _get_multiplier
from src import gp_bandit
from src import ml_helpers as mlh

def get_partition_scheduler(args):
    """
    Args:
        args : arguments from main.py
    Returns:
        callable beta_update function
    * callable has interface f(log_iw, args, **kwargs)
    * returns beta_id, or unchanged args.partition, by default
    Beta_update functions:
        *** MUST be manually specified here
        * given via args.partition_type and other method-specific args
        * should handle 0/1 endpoint insertion internally
        * may take args.K - 1 partitions as a result (0 is given)
    """

    schedule = args.schedule
    P = args.K - 1

    if schedule=='gp_bandit' or schedule=="gp"  or schedule=="gptv":
        return GP_bandits
    elif schedule=='rand':
        return rand_search
#    elif schedule=='gp_bandit_log':
#        return GP_bandits_log
    elif schedule in ['log', 'linear']:
        return beta_id
    elif schedule in ['moments']:
        return moments
#    elif schedule =='beta_gradient_descent':
#        return beta_gradient_descent
#    elif schedule =='beta_batch_gradient':
#        return beta_gradient_descent


def beta_id(model, args = None, **kwargs):
    """
    dummy beta update for static / unspecified partition_types
    """

    """
    f = bayes_quad.get_integrand_function(model, args)
    f_beta0=f(mlh.tensor(0,args)) # compute f(beta=0)
    args.f_beta0=f_beta0
    f=bayes_quad.get_integrand_function_subtract_beta0(model,args) #redefine the function using f(beta=0)
    Ytensor=f(args.partition)
    Y=Ytensor.data.cpu().numpy()

    try:
        oldY=np.load("Y_linear50")
    except:
        oldY=np.empty((0,1+args.K),float)
    oldY=np.vstack((oldY,np.reshape(Y,(1,-1))))
    np.save("Y_linear50",oldY)
    """
    #print(args.partition)
    return args.partition


def rand_search(model, args = None, **kwargs):
    """
    dummy beta update for static / unspecified partition_types
    """
    SearchSpace=np.asarray([0,1.0]*(args.K-1)).astype(float) # this is the search range of beta from 0-1
    SearchSpace=np.reshape(SearchSpace,(args.K-1,2))
    init_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
    init_X=np.around(init_X, decimals=4)
    init_X=np.append(0,init_X)
    init_X=np.append(init_X,1)

    init_X= np.sort(init_X)
    print(init_X)

    points=mlh.tensor(init_X,args)

    return points


def safe_step(partition, steps, min_val = 10**-6, max_val = 1.0, max_step = 0.01, adaptive = False, descent = False):
    ''' implement checks on beta values and sort if necessary after gradient descent step

        max_step : clips absolute value of steps = step_size * beta_derivative
        "adaptive" is very heuristic:
            scale down step size until at most 1 update is > max_step (maybe should be >= 1 since often only 1 big update)
    '''

    max_step = torch.ones_like(steps)*max_step
    min_val = torch.ones_like(partition)*min_val
    max_val = torch.ones_like(partition)*max_val;


    if adaptive:
        n_greater_than_max = torch.where(torch.abs(steps) > max_step, torch.ones_like(steps), torch.zeros_like(steps))
        while torch.sum(n_greater_than_max).item() > 1 :
            steps = steps*.1
            n_greater_than_max = torch.where(torch.abs(steps) > max_step, torch.ones_like(steps), torch.zeros_like(steps))
    else:
        steps = torch.where(torch.abs(steps) < max_step, steps, max_step*torch.sign(steps))

    partition = partition - steps.cpu() if descent else partition + steps.cpu()
    partition = torch.where(partition>=min_val, partition, min_val)
    partition = torch.where(partition<=max_val, partition, min_val)
    partition, _ = torch.sort(partition)
    return partition


def get_beta_derivative_diffs(model, args):
    '''
    Reads stored means in self.expectation_diffs and returns beta derivatives according to "async" or epoch update derivation ( i.e. variance term * 0 )
    Let f(β) = E_β log p(x,z)/q(z|x) , f'(β) = Var_β log p(x,z)/q(z|x)

    d TVO / dβ_k = f(β_k-1) - f(β_k ) - (β_k+1 - β_k) f'(β_k)

    only calculates for indices 1 => K-1 ( 0 and 1 are fixed )
    '''

    # if isinstance(model.expectation_diffs, mlh.AccumulatedDiff):
    #      # Not currently being used
    #     beta_deriv = torch.stack([(model.expectation_diffs.current[k-1] - model.expectation_diffs.current[k]) for k in range(1,model.expectation_diffs.current.shape[0]-1)])
    #     model.expectation_diffs.reset()
    # else:


    # expectation_diffs [k]= change in f(β_{k}),  derivative = change in expectation_diffs across beta

    # if isinstance(model.expectation_diffs, int):
    #     beta_deriv = torch.zeros_like(model.args.partition[1:-1].data)
    # else:
    # beta_deriv = torch.stack([(model.expectation_diffs[k-1] - model.expectation_diffs[k]) \
    #                  for k in range(1,model.expectation_diffs.shape[0]-1)])


    tvo_exps = model.exp_meter.mean.data
    tvo_vars = model.var_meter.mean.data
    last_exps = model.exp_last.data
    last_vars = model.var_last.data
    d_beta =  _get_multiplier(args.partition, 'left').squeeze().data

    # "two step" update => minus last
    beta_deriv = torch.stack( [ (tvo_exps[k-1] - tvo_exps[k]) - (last_exps[k-1] - last_exps[k] ) + d_beta[k]*(tvo_vars[k] -last_vars[k]) \
                              for k in range(1, tvo_exps.shape[0]-1) ]  )

    return beta_deriv

def get_beta_derivative_single(model, args):
    '''
    Reads stored means in self.expectation_diffs and returns beta derivatives according to "async" or epoch update derivation ( i.e. variance term * 0 )
    Let f(β) = E_β log p(x,z)/q(z|x) , f'(β) = Var_β log p(x,z)/q(z|x)

    d TVO / dβ_k = f(β_k-1) - f(β_k ) - (β_k+1 - β_k) f'(β_k)

    only calculates for indices 1 => K-1 ( 0 and 1 are fixed )
    '''

    # if isinstance(model.expectation_diffs, mlh.AccumulatedDiff):
    #      # Not currently being used
    #     beta_deriv = torch.stack([(model.expectation_diffs.current[k-1] - model.expectation_diffs.current[k]) for k in range(1,model.expectation_diffs.current.shape[0]-1)])
    #     model.expectation_diffs.reset()
    # else:


    # expectation_diffs [k]= change in f(β_{k}),  derivative = change in expectation_diffs across beta

    # if isinstance(model.expectation_diffs, int):
    #     beta_deriv = torch.zeros_like(model.args.partition[1:-1].data)
    # else:
    # beta_deriv = torch.stack([(model.expectation_diffs[k-1] - model.expectation_diffs[k]) \
    #                  for k in range(1,model.expectation_diffs.shape[0]-1)])

    tvo_exps = model.exp_meter.mean.data
    tvo_vars = model.var_meter.mean.data
    last_exps = model.exp_last.data
    last_vars = model.var_last.data
    d_beta =  _get_multiplier(args.partition, 'left').squeeze().data

    # "two step" update => minus last
    beta_deriv = torch.stack( [ (tvo_exps[k-1] - tvo_exps[k]) + d_beta[k]*tvo_vars[k] \
                              for k in range(1, tvo_exps.shape[0]-1) ]  )
    #  - (last_exps[k-1] - last_exps[k] )
    return beta_deriv



def beta_gradient_descent(model, args, cpu = True, diffs = False):
    '''
    perform manual gradient descent on beta
    - get_beta_derivative_no_var : returns dTVO / dbeta and resets beta tracking
    - safe_step clips updates

    recalculate = True is used for beta_batch_gradient
        - init_expectation = expectation before θ gradient descent update
        - expectation_diffs = expectation after θ gradient descent update
    '''

    if args.schedule=='beta_batch_gradient':
        # used to update per_batch
        # (re-calculate expectations, variance post θ-update)
        log_weight = model.elbo()
        tvo_exps = calc_exp(log_weight, args, all_sample_mean=True)
        tvo_vars = calc_var_given_betas(log_weight, args, all_sample_mean=True)

        model.exp_meter.step(tvo_exps.data)
        model.var_meter.step(tvo_vars.data)

        #if model.exp_last is None or model.var_last is None:
        #    model.exp_last = tvo_exps.data
        #    model.var_last = tvo_vars.data


    #model.expectation_diffs = tvo_exps.data - model.init_expectation
    elif model.exp_last is None or isinstance(model.exp_meter.mean, int):
        # unchanged for first epoch after burn-in on beta_gradient_descent
        return args.partition


    # also resets expectation differences
    if diffs:
        beta_derivatives = get_beta_derivative_diffs(model, args)
    else:
        beta_derivatives = get_beta_derivative_single(model, args)

    model.reset_track_beta()


    # Gradient Descent STEP
    sliced_partition = args.partition.data[1:-1]
    sliced_partition = sliced_partition.cpu() if cpu else sliced_partition

    new_partition = safe_step(sliced_partition, args.beta_step_size * beta_derivatives, max_step = args.max_beta_step, adaptive=args.adaptive_beta_step)

    # pad 0 and 1
    new_partition = torch.cat([torch.zeros_like(new_partition[0]).unsqueeze(0), new_partition,  torch.ones_like(new_partition[0]).unsqueeze(0)])

    print(args.partition)
    print("new partition ", new_partition)
    print("beta steps ", args.beta_step_size * beta_derivatives)


    return new_partition.cuda() if cpu else new_partition


def GP_bandits(model, args):
    points = gp_bandit.calculate_BO_points(model, args)
    K=len(points)
    points=mlh.tensor(points,args)
    print("==================================")
    print("K={} points={}".format(K,points))
    print("==================================")
    #args.K = K
    return points


def moments(model, args=None, **kwargs):
    args  = model.args if args is None else args
    start = 0
    stop  = 1
    threshold = 0.05

    if not args.per_sample and not args.per_batch:
        log_iw = get_total_log_weight(model, args, args.valid_S)
    else:
        log_iw = model.elbo()

    partitions = args.K-1
    targets = np.linspace(0.0, 1.0, num=args.K+1, endpoint=True)

    left  = calc_exp(log_iw, start, all_sample_mean= not(args.per_sample))
    right = calc_exp(log_iw, stop, all_sample_mean= not(args.per_sample))
    left  = torch.mean(left, axis = 0, keepdims = True) if args.per_batch else left
    right = torch.mean(right, axis = 0, keepdims= True) if args.per_batch else right
    moment_avg = right - left

    beta_result = []
    for t in range(len(targets)):
        if targets[t] == 0.0 or targets[t] == 1.0:
            beta_result.append(targets[t] * (torch.ones_like(log_iw[:,0]) if args.per_sample else 1) ) # zero if targets[t]=0
        else:
            target = targets[t]
            moment = left + target*moment_avg #for t in targets]

            start = torch.zeros_like(log_iw[:,0]) if args.per_sample else torch.zeros_like(left)
            stop = torch.ones_like(log_iw[:,0]) if args.per_sample else torch.ones_like(left)

            beta_result.append(_moment_binary_search(\
                    moment, log_iw, start = start, stop = stop, \
                        threshold=threshold, per_sample = args.per_sample))

    if args.per_sample: #or args.per_batch:
        beta_result = torch.cat([b.unsqueeze(1) for b in beta_result], axis=1).unsqueeze(1)
        beta_result, _ = torch.sort(beta_result, -1)
    else:
        beta_result = torch.cuda.FloatTensor(beta_result)

    return beta_result

def _moment_binary_search(target, log_iw, start=0, stop= 1, threshold = 0.1, recursion = 0, per_sample = False, min_beta = 0.001): #recursion = 0,
    beta_guess = .5*(stop+start)
    eta_guess = calc_exp(log_iw, beta_guess, all_sample_mean = not per_sample).squeeze()
    target = torch.ones_like(eta_guess)*(target.squeeze())
    start_ = torch.where( eta_guess <  target,  beta_guess, start)
    stop_ = torch.where( eta_guess >  target, beta_guess , stop)

    if torch.sum(  torch.abs( eta_guess - target) > threshold ).item() == 0:
        return beta_guess
    else:
        if recursion > 500:
            return beta_guess
        else:
            return _moment_binary_search(
                target,
                log_iw,
                start= start_,
                stop= stop_,
                recursion = recursion + 1,
                per_sample = per_sample)

def beta_id(model, args = None, **kwargs):
    """
    dummy beta update for static / unspecified partition_types
    """
    return args.partition
