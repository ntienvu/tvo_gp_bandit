from __future__ import division, print_function

import datetime
import errno
import json
import os
import pickle
import random
import shutil
import socket
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from joblib import Parallel, delayed
#from sklearn import metrics
from torch._six import inf


def safe_json_load(path):
    path = Path(path)
    res = {}
    if path.stat().st_size != 0:
        with open(path) as data_file:
            res = json.load(data_file)
    return res

def get_experiments_from_fs(path):
    path = Path(path)
    assert (path / '_sources/').exists(), f"Bad path: {path}"
    exps = {}
    dfs = []

    for job in path.glob("*"):
        if job.parts[-1] in ['_resources', '_sources']:
            continue
        job_id = job.parts[-1]

        run = safe_json_load(job / 'run.json')
        config = safe_json_load(job / 'config.json')
        metrics = safe_json_load(job / 'metrics.json')

        exps[job_id] = {**config, **run}

        if metrics:
            for metric, v in metrics.items():
                df = pd.DataFrame(v)
                df.index = pd.MultiIndex.from_product([[job_id], [metric], df.index], names=['_id', 'metric', 'index'])
                dfs += [df]

    exps = pd.DataFrame(exps).T
    exps.index.name = '_id'
    if dfs:
        df = pd.concat(dfs).drop('timestamps', axis=1)
    else:
        df = None
    return exps, df

def get_experiments_from_dir(path, observer_name="file_storage_observer"):
    path = Path(path)
    assert path.exists(), f'Bad path: {path}'
    exps = {}
    dfs = {}
    for p in path.rglob(observer_name):
        _id = str(p).replace(f"/{observer_name}", "")
        exp, df = get_experiments_from_fs(p)
        exps[_id] = exp
        if df is None:
            print(f"{p} returned empty df")
        else:
            dfs[_id] = df

    if exps and dfs:
        exps = pd.concat(exps.values(), keys=exps.keys()).droplevel(1)
        dfs = pd.concat(dfs.values(), keys=dfs.keys()).droplevel(1)

        exps.index.name = '_id'
        dfs.index.names = ['_id', 'metric', 'index']
    else:
        raise ValueError(f"results empty! path:{path}")

    return exps, dfs

def post_process(exp, df, CUTOFF_EPOCH=2000):
    print(f"{exp[exp.status == 'COMPLETED'].shape[0]} jobs completed")
    print(f"{exp[exp.status == 'RUNNING'].shape[0]} jobs timed out")
    print(f"{exp[exp.status == 'FAILED'].shape[0]} jobs failed")

    # Remove jobs that failed
    exp = exp[exp.status != 'FAILED']

    df = df[df.steps <= CUTOFF_EPOCH]

    # get values at last epoch
    results_at_cutoff = df[df.steps == CUTOFF_EPOCH].reset_index().pivot(index='_id', columns='metric',values='values')

    # join
    exp = exp.join(results_at_cutoff, how='outer')
    return exp, df


def process_dictionary_column(df, column_name):
    if column_name in df.columns:
        return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
    else:
        return df

def process_tuple_column(df, column_name, output_column_names):
    if column_name in df.columns:
        return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
    else:
        return df

def process_list_column(df, column_name, output_column_names):
    if column_name in df.columns:
        new = pd.DataFrame(df[column_name].values.tolist(), index=df.index, columns=output_column_names)
        old = df.drop(column_name, 1)
        return old.merge(new, left_index=True, right_index=True)
    else:
        return df


def get_data(path,isdir=False):
    #if 'file_storage_observer' in path or 'my_runs' in path or 'my_run_discrete' in path or "my_run" in path \
    #or 'my_run_discrete2' in path :
    #if isdir==False:
    try:
        exps, df = get_experiments_from_fs(Path(path))
    #else:
    except:
        exps, df = get_experiments_from_dir(Path(path))
    exps = exps[exps.status =='COMPLETED']
    #exps = exps[exps.schedule =='gp_bandit']
    exps = process_dictionary_column(exps, 'result')
    return exps, df

def get_baseline(name):
    exps = pd.read_csv(f"baseline_data/{name}_exps_10k.csv", index_col=0)
    df = pd.read_csv(f"baseline_data/{name}_df_10k.csv", index_col=[0,1])
    return exps, df

