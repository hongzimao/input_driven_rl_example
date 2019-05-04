import subprocess
from utils import create_folder_if_not_exists


procs = []
log_files = []
log_path = './results/logs/'
create_folder_if_not_exists(log_path)

# Regular state-dependent value network
prefix = 'regular_value_network'
log_file = open(log_path + prefix, 'w')
log_files.append(log_file)
p = subprocess.Popen(
    'python3 load_balance_actor_critic_train.py ' + \
    '--num_workers 10 --service_rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 '
    '--result_folder ./results/' + prefix + '/ ' + \
    '--model_folder ./results/parameters/' + prefix + '/',
    stdout=log_file, stderr=log_file, shell=True)
procs.append(p)

# Multi-value network (10 values)
prefix = '10_value_networks'
log_file = open(log_path + prefix, 'w')
log_files.append(log_file)
p = subprocess.Popen(
    'python3 load_balance_actor_multi_critic_train.py ' + \
    '--num_workers 10 --service_rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 '
    '--result_folder ./results/' + prefix + '/ ' + \
    '--model_folder ./results/parameters/' + prefix + '/',
    stdout=log_file, stderr=log_file, shell=True)
procs.append(p)

# clean up training process
for p in procs:
    p.wait()

# clean up log file handles
for f in log_files:
    f.close()