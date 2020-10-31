# Source code for Gaussian Process Bandit Optimization of the Thermodynamic Variational Objective

This uses [Sacred's](https://sacred.readthedocs.io/en/stable/command_line.html) command line interface. To see Sacred's options run

```
python run.py --help
```

To see tunable hyperparameters

```
python run.py print_config
```

which can be set using `with`:

```
python main.py with learning_task='continuous_vae' loss=tvo schedule='gp_bandit' S=50 K=5 epochs=1000
```

To save data to the filesystem, add a Sacred [FileStorageObserver](https://sacred.readthedocs.io/en/stable/observers.html)

```
python main.py with learning_task='continuous_vae' loss=tvo schedule='gp_bandit' S=50 K=5 epochs=1000 -F ./runs
```


The TVO loss is computed in [`get_thermo_loss_from_log_weight_log_p_log_q` in `losses.py`](https://github.com/vmasrani/tvo/blob/master/continuous_vae/losses.py#L149-L202). This function is identical to the one found in the discrete_vae directory.

The main training loop is in [`train` in `run.py`](https://github.com/vmasrani/tvo/blob/master/continuous_vae/run.py#138).