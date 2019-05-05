# Input-dependent Baseline
Input-dependent baseline for reducing the variance from external input processes.

Paper: https://openreview.net/forum?id=Hyg1G2AqtQ

### Example
- Regular A2C with state-dependent baseline on the load-balancing environment
```
python3 load_balance_actor_critic_train.py --num_workers 10 --service_rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 --result_folder ./results/regular_value_network/ --model_folder ./results/parameters/regular_value_network/
```

- A2C with multi-value baseline (10 value networks) on the load-balancing environment
```
python3 load_balance_actor_multi_critic_train.py --num_workers 10 --service_rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 --result_folder ./results/10_value_networks/ --model_folder ./results/parameters/10_value_networks/
```

- Monitor learning progress: Tensorboard in `./results/`; policy perforamnce on unseen traces plotted in `test_performance.png` in `./results/parameters/`.

### Dependencies
Python 3.6, Tensorflow 1.2.1, Numpy 1.14.5
