import numpy as np
from src import util
import logging
import torch
#from GPy.models import GPRegression
#from GPy.kern import RBF
#from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
#from emukit.quadrature.kernels import QuadratureRBF
#from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
#from emukit.quadrature.methods import VanillaBayesianQuadrature
#from emukit.quadrature.acquisitions import IntegralVarianceReduction
#from emukit.core.optimization import GradientAcquisitionOptimizer
#from emukit.core.parameter_space import ParameterSpace
from src.BOv import BayesOpt
import matplotlib.pyplot as plt
from src import ml_helpers as mlh
from src.BOv import unique_rows
import pickle
# Figure config
# NOT TUNABLE HYPERPARAMETERS
# (putting them here so I'm not tempted to tune them)

LEGEND_SIZE = 15
FIGURE_SIZE = (12, 8)
WINDOW = 5
K_MAX = 100
MIN_REL_CHANGE = 1e-3
MIN_ERR = 1e-3
LBM_GRID = np.linspace(-9, -0.1, 50)

emu_log = logging.getLogger("emukit")
emu_log.setLevel(logging.WARNING)

emu_gp = logging.getLogger("GP")
emu_gp.setLevel(logging.WARNING)


#def calculate_bq_points(model, args):
#    # Replace this with any integrand function
#    f = get_integrand_function(model, args)
#
#    # Initial partition (0, lbm, 1)
#    X = mlh.tensor((0, 10**args.bq_log_seed_point, 1.0), args)
#
#    Y = f(X)
#    emukit_method, optimizer = init_bq(X, Y)
#    points, est, k = auto_train_bq(X, Y, f, emukit_method, optimizer, args)
#
#    return points, est, k

def extract_X_Y_from_args(SearchSpace,args,T=None):
    # obtain X and Y by truncating the data in the time-varying setting
    # if the existing data is <3*arg.K, randomly generate X
    
    if T is None:
        T=args.truncation_threshold
    lenY=len(args.Y_ori)
    X=args.X_ori[max(0,lenY-T):,1:-1] # remove the first and last  column which is 0 and 1

    ur=unique_rows(X)
    if sum(ur)<(3*args.K): # random search to get initial data
        init_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        init_X=np.around(init_X, decimals=4)
        init_X=np.append(0,init_X)
        init_X=np.append(init_X,1)

        return np.sort(init_X),None

    if lenY%20==0:
        strPath="{:s}/save_X_Y_{:s}_S{:d}_K{:d}.p".format(str(args.artifact_dir), args.schedule, args.S, args.K)
        pickle.dump( [args.X_ori,args.Y_ori], open( strPath, "wb" ) )

    Y=np.reshape(args.Y_ori[max(0,lenY-T):],(-1,1))

    return X,Y

def append_Xori_Yori_from_args(args):
    # append the average logpx into Y
    # append the args.partition into X

    if args.len_terminated_epoch >0: # if we terminate the betas due to drip the epoch len is shorted
        average_y=np.mean(args.logtvopx_all[-args.len_terminated_epoch:])
    else:
        average_y=np.mean(args.logtvopx_all[-args.schedule_update_frequency:])
    args.average_y=np.append(args.average_y,average_y) # averaging the logpx over this window

    # error will happen at the first iteration when we add the first average_y into our data
    # this error is intentional, i will modify it by using a flag to indicate the first time
    if len(args.average_y)==1:
        print("ignore for the first time to save the first value of Y")
        return

    prev_y=args.average_y[-1] -args.average_y[-2]

    args.X_ori=np.vstack(( args.X_ori, np.reshape(format_input(args.partition),(1,args.K+1) )))
    args.Y_ori=np.append(args.Y_ori, prev_y)
    prev_X=np.round(args.X_ori[-1],decimals=4)
    print("X",prev_X,"Y",args.Y_ori[-1])

#def extract_X_Y_from_args_log(SearchSpace,args):
#    # obtain X and Y
#
#    T=args.truncation_threshold
#    lenY=len(args.Y_ori)
#
#    X=np.log10( args.X_ori[max(0,lenY-T):,1:-1]) # remove the first column which is 0
#
#    ur=unique_rows(X)
#    if sum(ur)< (3*args.K) or 'rand' in args.schedule: # random search to get initial data
#        init_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
#        init_X=np.around(10**init_X, decimals=4)
#        init_X=np.append(0,init_X)
#        init_X=np.append(init_X,1)
#
#        return np.sort(init_X),None
#
#    if args.X_ori.shape[0]%20==0: #save for analysis purpose
#        strPath="save_X_Y_{:s}_S{:d}_K{:d}.p".format(args.schedule,args.S,args.K)
#        pickle.dump( [args.X_ori,args.Y_ori], open( strPath, "wb" ) )
#
#    Y=np.reshape(args.Y_ori[max(0,lenY-T):],(-1,1))
#
#    return X,Y


def calculate_BO_points(model,args):

    # process the input X and output Y from args
    append_Xori_Yori_from_args(args)

    SearchSpace=np.asarray([args.bandit_beta_min,args.bandit_beta_max]*(args.K-1)).astype(float) # this is the search range of beta from 0-1
    SearchSpace=np.reshape(SearchSpace,(args.K-1,2))

    if args.K==2:
        SearchSpace[0,1]=0.7
    else:
        ll=np.linspace(0,args.bandit_beta_max,args.K) # to discourage selecting 1
        for kk in range(args.K-1):
            SearchSpace[kk,1]=ll[kk+1]



    # truncate the time-varying data if neccessary
    # if dont have enough data -> randomly generate data
    if args.schedule=="gp": # non timevarying
        X,Y=extract_X_Y_from_args(SearchSpace,args,T=len(args.Y_ori))
    else:   # time varying     
        X,Y=extract_X_Y_from_args(SearchSpace,args)
        
    if Y is None:
        return X

    # augment the data with artificial observations all zeros and all ones
    x_all_zeros=np.reshape(np.asarray([args.bandit_beta_min]*(args.K-1)),(1,-1))
    x_all_ones=np.reshape(np.asarray([args.bandit_beta_max]*(args.K-1)),(1,-1))

    worse_score=np.min(Y)

    X=np.vstack((X,x_all_zeros))
    X=np.vstack((X,x_all_ones))

    Y=np.vstack((Y,np.asarray(worse_score)))
    Y=np.vstack((Y,np.asarray(worse_score)))


    # perform GP bandit
    if args.schedule=="gp_bandit":
        myBO=BayesOpt(func=None,SearchSpace=SearchSpace)
    elif args.schedule=="tvgp" or args.schedule=="gp": # TV but not permutation invariant
        myBO=BayesOpt(func=None,SearchSpace=SearchSpace,GPtype="vanillaGP")    
    else:
        print("please change ",args.schedule)
        
    
    myBO.init_with_data(X,Y)

    # beta is selected from here
    new_X=myBO.select_next_point()[1]

    # sorting
    new_X=np.round(new_X,decimals=4)
    new_X = np.append(np.append(0,np.sort(new_X)), 1)
    print(new_X)

    temp_new_X=np.unique(new_X)

    if np.array_equal(temp_new_X, [0, args.bandit_beta_min, 1]) or \
        np.array_equal(temp_new_X, [0, 1]) :#0.01 is due to the search bound
        rand_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        return np.append(np.append(0,np.sort(rand_X)), 1)
    else:
        return new_X

    
def calculate_BO_points_vanillaGP(model,args): # this is used as a baseline with vanilla GP

    # process the input X and output Y from args
    append_Xori_Yori_from_args(args)

    SearchSpace=np.asarray([args.bandit_beta_min,args.bandit_beta_max]*(args.K-1)).astype(float) # this is the search range of beta from 0-1
    SearchSpace=np.reshape(SearchSpace,(args.K-1,2))

    if args.K==2:
        SearchSpace[0,1]=0.7
    else:
        ll=np.linspace(0,args.bandit_beta_max,args.K) # to discourage selecting 1
        for kk in range(args.K-1):
            SearchSpace[kk,1]=ll[kk+1]

    X,Y=extract_X_Y_from_args(SearchSpace,args,T=len(args.Y_ori)) # this is non timevarying GP, takes all data
    if Y is None:
        return X

    myBO=BayesOpt(func=None,SearchSpace=SearchSpace,GPtype="vanillaGP")
    myBO.init_with_data(X,Y)

    # beta is selected from here
    new_X=myBO.select_next_point()[1]
    new_X=np.round(new_X,decimals=4)

    new_X = np.append(np.append(0,np.sort(new_X)), 1)
    print(new_X)

    temp_new_X=np.unique(new_X)

    if np.array_equal(temp_new_X, [0, args.bandit_beta_min, 1]) or \
        np.array_equal(temp_new_X, [0, 1]) :#0.01 is due to the search bound
        rand_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        return np.append(np.append(0,np.sort(rand_X)), 1)
    else:
        return new_X
    

#def get_derivative_points(X,log_weight,args):
#    Xdrv=[]
#    derivatives=[]
#    for ii in range(X.shape[0]-1): # placing init_K-1 initial derivative observations between two points
#        delta_x=np.abs(X[ii,0]-X[ii+1,0]) # different in x
#        if delta_x<1e-2:
#            continue
#        new_point=delta_x/2+X[ii,0]
#        Xdrv=Xdrv+[new_point] # set the derivative point at the middle of two points
#
#        der = util.calc_var_given_betas(log_weight, args, all_sample_mean=True, betas = new_point)
#        derivatives=derivatives+[round(np.asscalar(format_input(der)),3)]
#    print(derivatives)
#    Xdrv=np.reshape(np.asarray(Xdrv),(-1,1)) # convert Xdrv to [M*d] where d=1
#    derivatives=np.reshape(np.asarray(derivatives),(-1,1)) # convert derivatives to [M*1]
#    return Xdrv, derivatives


#def calculate_bqd_points(model, args): #cal BQ with derivative
#    # we extract log_weight two times! The above line also extracts this. So please optimise!
#    # this log_weight is used to estimate the derivative
#    log_weight = util.get_total_log_weight(model, args, args.valid_S)
#
#    args.log_weight=log_weight
#
#    # Replace this with any integrand function
#
#    X,Y,f,SearchSpace=init_bqd(model,args)
#
#    #extracting the derivative points
#    Xdrv,derivatives= get_derivative_points(X,log_weight,args)
#    #Xdrv,derivatives= get_derivative_points_middle(X,log_weight,args)
#    Y=np.reshape(np.asarray(Y),(-1,1)) # convert Y to [K*1]
#
#    myGPdrv=GaussianProcessDerivative(SearchSpace)
#    myGPdrv.fit_drv(X,Y,Xdrv,derivatives,drv_index=0)
#    myGPdrv.auto_train_bq(f,MaxK=15)
#
#    est=myGPdrv.integral()[1]
#    points=myGPdrv.X.ravel()
#
#    points=np.asarray([0,0.1,1])
#    #points=np.asarray([0.0000,     0.0001,     0.0017,     0.0210,     0.0724,     0.1434,
#           # 0.1741,     0.5114,     0.8678,     0.9030,     0.9377,     0.9911,1.0000])
#    #return points, 0, len(points)
#
#    #print("[GP with drv] estimated integral is",est) # this function is useful to return the integral
#    return np.sort(points), est, len(points)


#def init_bqd(model,args):
#    log_weight = util.get_total_log_weight(model, args, args.valid_S)
#    args.log_weight=log_weight
#
#    # Replace this with any integrand function
#    # I just want to estimate f(beta=0) for normalising purpose.
#    # This is better way to estimate and do it instead of two repeating calls.
#
#    init_K=3
#    SearchSpace=np.asarray([[0,1]]) # this is the search range of beta from 0-1
#
#    if "loss" in args.schedule:
#        f=get_integrand_function_from_loss(model,args)
#    elif "log" in args.schedule:
#        f = get_integrand_function(model, args)
#        f_beta0=f(mlh.tensor(0,args)) # compute f(beta=0)
#        args.f_beta0=f_beta0
#        f=get_integrand_function_subtract_beta0_log(model,args) #redefine the function using f(beta=0)
#        SearchSpace=np.asarray([[-1,0]]) # this is the search range of beta from 0-1
#    else:
#        f = get_integrand_function(model, args)
#        f_beta0=f(mlh.tensor(0,args)) # compute f(beta=0)
#        args.f_beta0=f_beta0
#        f=get_integrand_function_subtract_beta0(model,args) #redefine the function using f(beta=0)
#
#    # Initial partition (0,1)
#    X=np.linspace(SearchSpace[0,0],SearchSpace[0,1],init_K)
#    X=np.reshape(X,(-1,1))
#    Xtensor = mlh.tensor(X, args)
#
#    if "log" in args.schedule:
#        Ytensor = [f(10**Xtensor[ii]) for ii in range(init_K)]
#    else:
#        Ytensor = [f(Xtensor[ii]) for ii in range(init_K)]
#
#    Y=[np.asscalar(Ytensor[ii].data.cpu().numpy()) for ii in range(init_K)] # convert to numpy
#    Y=np.reshape(np.asarray(Y),(-1,1)) # convert Y to [K*1]
#
#    return X,Y,f,SearchSpace


def format_input(*data):
    output = []
    for d in data:
        if torch.is_tensor(d):
            d = d.cpu().numpy()
        if d.ndim == 1:
            d = np.expand_dims(d, 1)
        output.append(d)

    return output[0] if len(output) == 1 else output


def plot_variance(ivr_acquisition):
    x_plot = np.linspace(0, 1, 300)[:, None]
    ivr_plot = ivr_acquisition.evaluate(x_plot)

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x_plot, (ivr_plot - np.min(ivr_plot)) / (np.max(ivr_plot) - np.min(ivr_plot)),
             "green", label="integral variance reduction")

    plt.legend(loc=0, prop={'size': LEGEND_SIZE})
    plt.xlabel(r"$x$")
    plt.ylabel(r"$acquisition(x)$")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1.04)
    plt.show()


#def init_bq(X, Y, integral_bounds=[(0, 1)]):
#    X, Y = format_input(X, Y)
#    gpy_model = GPRegression(X=X, Y=Y, kernel=RBF(input_dim=X.shape[1],
#                                                  lengthscale=0.5,
#                                                  variance=1.0))
#
#    # Kernals and stuff
#    emukit_rbf    = RBFGPy(gpy_model.kern)
#    emukit_qrbf   = QuadratureRBFLebesgueMeasure(emukit_rbf, integral_bounds=integral_bounds)
#    emukit_model  = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
#    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model,X=X,Y=Y)
#
#    space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
#    optimizer = GradientAcquisitionOptimizer(space)
#
#    #space = ParameterSpace(emukit_method.integral_parameters)
#    #optimizer = GradientAcquisitionOptimizer(space)
#    return emukit_method, optimizer


#def auto_calculate_bq_points(model, args):
#    f = get_integrand_function(model, args)
#
#    best_meter = mlh.BestMeter()
#
#    for lbm in LBM_GRID:
#        print("Running lbm = {} ...".format(round(lbm, 3)))
#        X = mlh.tensor((0, 10**lbm, 1.0), args)
#        Y = f(X)
#        emukit_method, optimizer = init_bq(X, Y)
#        points, est, k = auto_train_bq(X, Y, f, emukit_method, optimizer, args)
#        best_meter.step(est, (points, lbm, k))
#
#    best_points, best_lbm, best_k = best_meter.best_obj
#    best_est = best_meter.best
#    return best_points, best_est, best_k, best_lbm


#def auto_train_bq(X, Y, f, emukit_method, optimizer, args):
#    X, Y = format_input(X, Y)
#
#    maxed_out = True
#    mean_avg, err_avg = mlh.MovingAverageMeter('mean', ':.15f', window=WINDOW), mlh.MovingAverageMeter('err', ':.15f', window=WINDOW)
#
#    for k in range(K_MAX):
#        integral_mean, integral_variance = emukit_method.integrate()
#        err = 2 * np.sqrt(integral_variance)
#
#        mean_avg.step(integral_mean)
#        err_avg.step(err)
#
#        ivr_acquisition = IntegralVarianceReduction(emukit_method)
#
#        #plot_variance(ivr_acquisition)
#
#        x_new, _ = optimizer.optimize(ivr_acquisition)
#
#        x_new = mlh.tensor(x_new, args)
#        y_new = f(x_new)
#
#        X = np.append(X, format_input(x_new), axis=0)
#        Y = np.append(Y, format_input(y_new), axis=0)
#
#
#        emukit_method.set_data(X, Y)
#
#        # check break condition
#        if (abs(mean_avg.relative_change) < MIN_REL_CHANGE) and (abs(err_avg.val) < MIN_ERR):
#            maxed_out = False
#            break
#
#    if maxed_out:
#        print("################################################")
#        print(f"---------------- Warning --------------------- ")
#        print(f"Inner loop failed to converge after {K_MAX} iterations ###")
#        print(f"mean_avg: {mean_avg}")
#        print(f"err_avg: {err_avg}")
#        print("################################################")
#
#    return mlh.tensor(np.sort(X.flatten()), args), mean_avg.val, k


#def get_integrand_function_subtract_beta0(model, args):
#
#    try: # for BDQ we already have log_weight, dont need to extract it again
#        log_weight=args.log_weight
#    except: # for other approaches, we need to extract it
#        log_weight = util.get_total_log_weight(model, args, args.valid_S)
#
#    def f(X):
#        partition = mlh.tensor(X,args)
#        heated_log_weight = log_weight.unsqueeze(-1) * partition
#
#        heated_normalized_weight = mlh.exponentiate_and_normalize(heated_log_weight, dim=1)
#        Y = torch.sum(heated_normalized_weight * log_weight.unsqueeze(-1), dim=1).mean(0)
#
#        return Y-args.f_beta0
#    return f

#def get_integrand_function_subtract_beta0_log(model, args):
#
#    try: # for BDQ we already have log_weight, dont need to extract it again
#        log_weight=args.log_weight
#    except: # for other approaches, we need to extract it
#        log_weight = util.get_total_log_weight(model, args, args.valid_S)
#
#    def f(X): # we take 10**X due to the log space
#        partition = mlh.tensor(10**X,args)
#        heated_log_weight = log_weight.unsqueeze(-1) * partition
#
#        heated_normalized_weight = mlh.exponentiate_and_normalize(heated_log_weight, dim=1)
#        Y = torch.sum(heated_normalized_weight * log_weight.unsqueeze(-1), dim=1).mean(0)
#
#        return Y-args.f_beta0
#    return f

#def get_integrand_function(model, args):
#
#    try: # for BDQ we already have log_weight, dont need to extract it again
#        log_weight=args.log_weight
#    except: # for other approaches, we need to extract it
#        log_weight = util.get_total_log_weight(model, args, args.valid_S)
#
#    def f(X):
#        partition = mlh.tensor(X,args)
#        heated_log_weight = log_weight.unsqueeze(-1) * partition
#
#        heated_normalized_weight = mlh.exponentiate_and_normalize(heated_log_weight, dim=1)
#        Y = torch.sum(heated_normalized_weight * log_weight.unsqueeze(-1), dim=1).mean(0)
#
#        return Y
#    return f

def get_cov_function(model, args):

    try: # for BDQ we already have log_weight, dont need to extract it again
        log_weight=args.log_weight
    except: # for other approaches, we need to extract it
        log_weight = util.get_total_log_weight(model, args, args.valid_S)

    def f(X):
        der = util.calc_var_given_betas(log_weight, args, all_sample_mean=True,
                                        betas = X)
        return der
    return f
