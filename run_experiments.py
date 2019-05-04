import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import matplotlib.pyplot as plt
import environments as envs
from utils import *
from param import *
from load_balance.heuristic_agents import *
from actor_agent import *


def main():

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)

    # different agents for different environments
    if args.env == 'load_balance':
        schemes = ['shortest_processing_time', 'learn']
    else:
        print 'Schemes for ' + args.env + ' does not exist'
        exit(1)

    # tensorflow session
    sess = tf.Session()

    # store results
    all_performance = {scheme: [] for scheme in schemes}

    # create environment
    env = envs.make(args.env)

    # initialize all agents
    agents = {}
    for scheme in schemes:

        if scheme == 'learn':
            agents[scheme] = ActorAgent(sess)
            # saver for loading trained model
            saver = tf.train.Saver(max_to_keep=args.num_saved_models)
            # initialize parameters
            sess.run(tf.global_variables_initializer())
            # load trained model
            if args.saved_model is not None:
                saver.restore(sess, args.saved_model)

        elif scheme == 'leat_work':
            agents[scheme] = LeastWorkAgent()

        elif scheme == 'shortest_processing_time':
            agents[scheme] = ShortestProcessingTimeAgent()

        else:
            print 'invalid scheme', scheme
            exit(1)

    # run testing experiments
    for ep in xrange(args.num_ep):

        for scheme in schemes:

            # reset the environment with controlled seed
            env.set_random_seed(ep)
            env.reset()

            # pick agent
            agent = agents[scheme]

            # store total reward
            total_reward = 0

            # -- run the environment --
            t1 = time.time()

            state = env.observe()
            done = False

            while not done:
                action = agent.get_action(state)
                state, reward, done = env.step(action)
                total_reward += reward

            t2 = time.time()
            print 'Elapsed', scheme, t2 - t1, 'seconds'

            all_performance[scheme].append(total_reward)

        # plot job duration cdf
        fig = plt.figure()

        title = 'average: '

        for scheme in schemes:
            x, y = compute_CDF(all_performance[scheme])
            plt.plot(x, y)

            title += ' ' + scheme + ' '
            title += '%.2f' % np.mean(all_performance[scheme])

        plt.xlabel('Total reward')
        plt.ylabel('CDF')
        plt.title(title)
        plt.legend(schemes)

        fig.savefig(args.result_folder + \
            args.env + '_all_performance.png')
        plt.close(fig)

        # save all job durations
        np.save(args.result_folder + \
            args.env + '_all_performance.npy', \
            all_performance)

    sess.close()


if __name__ == '__main__':
    main()
