# import gym
from load_balance.env import LoadBalanceEnvironment
# from abr.env import ABREnvironment


def make(env_name):
    if env_name == 'load_balance':
        env = LoadBalanceEnvironment()

    # elif env_name == 'abr':
    #     env = ABREnvironment()

    # # toy environment
    # elif env_name == 'InvertedPendulum-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Swimmer-v2':
    #     env = gym.make(env_name)

    # # mujoco environments
    # elif env_name == 'Ant-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'HalfCheetah-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Hopper-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Walker2d-v2':
    #     env = gym.make(env_name)

    # # disturbed mujoco environments
    # elif env_name == 'AntDist-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'HalfCheetahDist-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'HopperDist-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Walker2dDist-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'HalfCheetahBlocks-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'ArmDist-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'ArmMovingTarget-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Walker2dDistDoubleAction-v2':
    #     env = gym.make(env_name)

    # elif env_name == 'Walker2dDistDoubleActionDist-v2':
    #     env = gym.make(env_name)

    else:
        print('Environment ' + env_name + ' is not supported')
        exit(1)

    return env