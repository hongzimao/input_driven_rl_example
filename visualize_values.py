import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib
matplotlib.use('agg')
import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from param import *
from environment import *
from actor_agent import *
from job_generator import *
from average_reward import *
from compute_baselines import *
from tensorboard_summaries import *


def collect_values(agent_id, param_queue, job_queue, value_queue):

    np.random.seed(args.seed)  # for environment
    tf.set_random_seed(agent_id)  # for model evolving

    sess = tf.Session()

    # set up actor agent for training
    actor_agent = ActorAgent(sess)

    # synchronize model parameters
    actor_params = param_queue.get()
    actor_agent.set_params(actor_params)

    # collect experiences
    for ep in xrange(args.num_ep):

        # get streaming job sequence
        stream_jobs, service_rates = job_queue.get()

        # set up envrionemnt
        env = Environment(len(stream_jobs), stream_jobs, service_rates)

        # set up training storage
        batch_reward, batch_wall_time = [], []

        # run experiment
        state = env.observe()
        done = False

        while not done:

            # decompose state (for storing infomation)
            workers, job, curr_time = state

            inputs = np.zeros([1, args.num_workers + 1])
            for worker in workers:
                inputs[0, worker.worker_id] = \
                    sum(j.size for j in worker.queue) / \
                    args.job_size_max / 10.0  # normalization
            inputs[0, -1] = job.size / args.job_size_max  # normalization

            # draw an action
            action = actor_agent.predict(inputs)[0]

            # store wall time
            batch_wall_time.append(curr_time)

            # interact with environment
            state, reward, done = env.step(action)

            # scale reward for training
            reward /= args.reward_scale

            # store reward
            batch_reward.append(reward)

        # store final time
        batch_wall_time.append(env.wall_time.curr_time)

        # report rewards to master agent
        value_queue.put([batch_reward, batch_wall_time])

    sess.close()


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    param_queues = [mp.Queue(1) for _ in xrange(args.num_agents)]
    job_queues = [mp.Queue(1) for _ in xrange(args.num_agents)]
    value_queues = [mp.Queue(1) for _ in xrange(args.num_agents)]

    # set up training agents
    agents = []
    for i in xrange(args.num_agents):
        agents.append(mp.Process(target=collect_values, args=(
            i, param_queues[i], job_queues[i], value_queues[i])))

    # start training agents
    for i in xrange(args.num_agents):
        agents[i].start()

    # set up central session
    sess = tf.Session()

    # set up actor agent in master thread
    actor_agent = ActorAgent(sess)

    # initialize model parameters
    sess.run(tf.global_variables_initializer())

    # set up logging processes
    saver = tf.train.Saver(max_to_keep=args.num_saved_models)

    # load trained model
    if args.saved_model is not None:
        saver.restore(sess, args.saved_model)

    # synchronize the model parameters for each agent
    actor_params = actor_agent.get_params()

    for i in xrange(args.num_agents):
        param_queues[i].put(actor_params)

    # initialize worker service rates
    if args.service_rates is not None:
        assert len(args.service_rates) == args.num_workers
        service_rates = args.service_rates
    else:
        service_rates = [np.random.uniform(
            args.service_rate_min, args.service_rate_max) \
            for _ in xrange(args.num_workers)]

    # ---- visualize some values ----
    for ep in xrange(args.num_ep):
        print 'collection epoch', ep

        stream_jobs = generate_jobs(int(args.num_stream_jobs))

        # send out parameters to training agents
        for i in xrange(args.num_agents):
            job_queues[i].put([stream_jobs, service_rates])

        # storage for advantage computation
        all_reward, all_wall_time, all_diff_time = [], [], []

        # store average reward for computing differential rewards
        avg_reward_calculator = AveragePerStepReward(args.average_reward_storage)

        t1 = time.time()

        # update average reward
        for i in xrange(args.num_agents):

            batch_reward, batch_wall_time = value_queues[i].get()

            batch_diff_time = np.array(batch_wall_time[1:]) - np.array(batch_wall_time[:-1])

            avg_reward_calculator.add_list_filter_zero(batch_reward, batch_diff_time)

            all_reward.append(batch_reward)

            # for diff reward
            all_wall_time.append(batch_wall_time[:-1])
            all_diff_time.append(batch_diff_time)

        t2 = time.time()
        print 'got reward info from workers', t2 - t1, 'seconds'

        # compute differential reward
        all_diff_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in xrange(args.num_agents):
            diff_reward = np.array([r - avg_per_step_reward * t for \
                (r, t) in zip(all_reward[i], all_diff_time[i])])

            diff_cum_reward = discount(diff_reward, args.gamma)
            all_diff_cum_reward.append(diff_cum_reward)

        # compute wall time based baseline
        # baselines = get_ployfit_baseline(all_diff_cum_reward, all_wall_time)
        baselines = get_piecewise_linear_fit_baseline(all_diff_cum_reward, all_wall_time)

        # visualize value trajectories
        fig = plt.figure()

        for i in xrange(args.num_agents):
            plt.plot(all_wall_time[i], all_diff_cum_reward[i], 'b', alpha=0.8)
            plt.plot(all_wall_time[i], baselines[i], 'black', alpha=0.8)

        plt.xlabel('Time')
        plt.ylabel('Differential value')

        plt.tight_layout()
        plt.savefig(args.result_folder + \
            'value_visualization_' + str(ep) + '.png')
        plt.close(fig)

    sess.close()


if __name__ == '__main__':
    main()
