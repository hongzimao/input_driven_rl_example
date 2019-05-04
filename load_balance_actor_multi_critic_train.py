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
import environments as envs
from utils import *
from param import *
from load_balance_actor_agent import *
from critic_agent import *
from average_reward import *
from tensorboard_summaries import *
from actor_critic_test import run_test


def training_agent(agent_id, params_queue, reward_queue, adv_queue, gradient_queue):
    np.random.seed(args.seed)  # for environment
    tf.set_random_seed(agent_id)  # for model evolving

    sess = tf.Session()

    # set up actor agent for training
    actor_agent = ActorAgent(sess)
    critic_agent = CriticAgent(sess,
        input_dim=args.num_workers + 2)

    # set up envrionemnt
    env = envs.make(args.env)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, critic_params, entropy_weight, model_idx) = \
            params_queue.get()

        # synchronize model parameters
        actor_agent.set_params(actor_params)
        critic_agent.set_params(critic_params)

        # reset environment
        env.set_random_seed(model_idx)
        env.reset()

        # set up training storage
        batch_inputs, batch_act_vec, batch_wall_time, batch_reward = \
            [], [], [], []

        # run experiment
        state = env.observe()
        done = False

        while not done:

            # decompose state (for storing infomation)
            workers, job, curr_time = state

            inputs = np.zeros([1, args.num_workers + 1])
            for worker in workers:
                inputs[0, worker.worker_id] = \
                    min(sum(j.size for j in worker.queue) / \
                    args.job_size_norm_factor / 5.0,  # normalization
                    20.0)
            inputs[0, -1] = min(job.size / \
                args.job_size_norm_factor, 10.0)  # normalization

            # draw an action
            action = actor_agent.predict(inputs)[0]

            # store input and action
            batch_inputs.append(inputs)

            act_vec = np.zeros([1, args.num_workers])
            act_vec[0, action] = 1

            batch_act_vec.append(act_vec)

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

        # compute all values
        value_inputs = np.zeros([len(batch_inputs), args.num_workers + 2])
        for i in range(len(batch_inputs)):
            value_inputs[i, :-1] = batch_inputs[i]
            value_inputs[i, -1] = batch_wall_time[i] / float(batch_wall_time[-1])
        batch_values = critic_agent.predict(value_inputs)

        # summarize more info for master agent
        unfinished_jobs = sum(len(worker.queue) for worker in env.workers)
        unfinished_jobs += sum(worker.curr_job is not None for worker in env.workers)

        finished_work = sum(j.size for j in env.finished_jobs)
        unfinished_work = 0
        for worker in env.workers:
            for j in worker.queue:
                unfinished_work += j.size
            if worker.curr_job is not None:
                unfinished_work += worker.curr_job.size

        average_job_duration = np.mean([
            j.finish_time - j.arrival_time for j in env.finished_jobs])

        # report rewards to master agent
        reward_queue.put([
            batch_reward, np.array(batch_values), batch_wall_time,
            len(env.finished_jobs), unfinished_jobs,
            finished_work, unfinished_work, average_job_duration])

        # get advantage term
        batch_adv, batch_actual_value = adv_queue.get()

        # conpute gradient
        actor_gradient, loss = actor_agent.compute_gradients(
            batch_inputs, batch_act_vec, batch_adv, entropy_weight)
        critic_gradient, _ = critic_agent.compute_gradients(
            value_inputs, batch_actual_value)

        # send back gradients
        gradient_queue.put([actor_gradient, critic_gradient, loss])

    sess.close()


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=training_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # set up central session
    sess = tf.Session()

    # set up actor agent in master thread
    actor_agent = ActorAgent(sess)

    # set up multiple critic agent in master thread
    critic_agents = []
    for i in range(args.num_models):
        critic_agents.append(CriticAgent(
            sess, input_dim=args.num_workers + 2,
            scope='critic_agent_' + str(i)))
        progress_bar(i + 1, args.num_models,
            status='create critic models ' + \
            str(i + 1) + '/' + str(args.num_models))

    # initialize model parameters
    sess.run(tf.global_variables_initializer())

    # set up logging processes
    saver = tf.train.Saver(max_to_keep=args.num_saved_models)
    summary_ops, summary_vars = build_load_balance_tf_summaries()
    writer = tf.summary.FileWriter(
        args.result_folder + args.model_folder + \
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # load trained model
    if args.saved_model is not None:
        saver.restore(sess, args.saved_model)

    # initialize training parameters
    entropy_weight = args.entropy_weight_init
    reset_prob = args.reset_prob
    num_stream_jobs = args.num_stream_jobs

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(args.average_reward_storage)

    # Performance monitoring
    all_iters = []
    all_perf = [[], [], []]  # mean - std, mean, mean + std

    # ---- start training process ----
    for ep in range(1, args.num_ep):

        print('training epoch', ep)

        model_indices = np.random.choice(args.num_models, args.num_agents)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # send out parameters to training agents
        for i in range(args.num_agents):
            # get model index
            m = model_indices[i]
            # get corresponding critic parameter
            critic_params = critic_agents[m].get_params()
            # assign worker for the stream
            params_queues[i].put(
                [actor_params, critic_params, entropy_weight, m])

        # storage for advantage computation
        all_reward, all_values, all_wall_time, all_diff_time, all_eps_duration, \
            all_eps_finished_jobs, all_eps_unfinished_jobs, \
            all_eps_finished_work, all_eps_unfinished_work, \
            all_average_job_duration = \
            [], [], [], [], [], [], [], [], [], []

        t1 = time.time()

        # update average reward
        for i in range(args.num_agents):

            batch_reward, batch_values, batch_wall_time, \
                eps_finished_jobs, eps_unfinished_jobs, \
                eps_finished_work, eps_unfinished_work, \
                average_job_duration = \
                reward_queues[i].get()

            batch_diff_time = np.array(batch_wall_time[1:]) - np.array(batch_wall_time[:-1])

            avg_reward_calculator.add_list_filter_zero(batch_reward, batch_diff_time)

            all_reward.append(batch_reward)
            all_values.append(batch_values)

            # for diff reward
            all_wall_time.append(batch_wall_time[:-1])
            all_diff_time.append(batch_diff_time)

            # for tensorboard
            all_eps_duration.append(batch_wall_time[-1])
            all_eps_finished_jobs.append(eps_finished_jobs)
            all_eps_unfinished_jobs.append(eps_unfinished_jobs)
            all_eps_finished_work.append(eps_finished_work)
            all_eps_unfinished_work.append(eps_unfinished_work)
            all_average_job_duration.append(average_job_duration)

        t2 = time.time()
        print('got reward info from workers', t2 - t1, 'seconds')

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                    (r, t) in zip(all_reward[i], all_diff_time[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                    (r, t) in zip(all_reward[i], all_diff_time[i])])

            cum_reward = discount(rewards, args.gamma)
            all_cum_reward.append(cum_reward)

        # give worker back the advantage
        for i in range(args.num_agents):
            all_cum_reward[i] = np.reshape(all_cum_reward[i],
                [len(all_cum_reward[i]), 1])
            batch_adv = all_cum_reward[i] - all_values[i]
            adv_queues[i].put([batch_adv, all_cum_reward[i]])

        t3 = time.time()
        print('baseline ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (gradients, critic_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(gradients)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / \
                float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

            critic_agents[model_indices[i]].apply_gradients(
                critic_gradient, args.lr_rate)

        actor_agent.apply_gradients(aggregate_gradients(actor_gradients), args.lr_rate)

        t4 = time.time()
        print('worker send back and apply gradients', t4 - t3, 'seconds')

        print('average reward', avg_per_step_reward * -args.reward_scale)

        summary_str = sess.run(summary_ops, feed_dict={
            summary_vars[0]: np.mean(all_action_loss),
            summary_vars[1]: np.mean(all_entropy),
            summary_vars[2]: np.mean(all_value_loss),
            summary_vars[3]: np.mean([b.shape[0] for b in all_values]),
            summary_vars[4]: avg_per_step_reward * -args.reward_scale,
            summary_vars[5]: np.mean([r[0] for r in all_cum_reward]),
            summary_vars[6]: np.mean([t for t in all_eps_duration]),
            summary_vars[7]: entropy_weight,
            summary_vars[8]: reset_prob,
            summary_vars[9]: num_stream_jobs,
            summary_vars[10]: np.mean([t >= 0 for t in all_eps_duration]),
            summary_vars[11]: np.mean(all_eps_finished_jobs),
            summary_vars[12]: np.mean(all_eps_unfinished_jobs),
            summary_vars[13]: np.mean(all_eps_finished_work),
            summary_vars[14]: np.mean(all_eps_unfinished_work),
            summary_vars[15]: np.mean(all_average_job_duration)
        })

        writer.add_summary(summary_str, ep)
        writer.flush()

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
            args.entropy_weight_min, args.entropy_weight_decay)

        if ep % args.model_save_interval == 0:
            saver.save(sess, args.model_folder + "model_ep_" + str(ep) + ".ckpt")
            # perform testing
            test_result = run_test(actor_agent)
            # plot testing
            all_iters.append(ep)
            test_mean = np.mean(test_result)
            test_std = np.std(test_result)
            all_perf[0].append(test_mean - test_std)
            all_perf[1].append(test_mean)
            all_perf[2].append(test_mean + test_std)
            fig = plt.figure()
            plt.fill_between(all_iters, all_perf[0], all_perf[2], alpha=0.5)
            plt.plot(all_iters, all_perf[1])
            plt.xlabel('iteration')
            plt.ylabel('Total testing reward')
            plt.tick_params(labelright=True)
            fig.savefig(args.model_folder + 'test_performance.png')
            np.save(args.model_folder + 'test_performance.npy', all_perf)
            plt.close(fig)

    sess.close()


if __name__ == '__main__':
    main()
