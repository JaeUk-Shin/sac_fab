import time
from datetime import datetime
import os
import csv
import argparse
import random
import torch
import gym
import gym_lifter
from sac_agent import SACAgent
from utils import get_env_spec, set_log_dir


def run_sac(
            env_id,
            num_ep=2e3,
            T=300,
            eval_interval=2000,
            start_train=10000,
            train_interval=50,
            buffer_size=1e6,
            fill_buffer=20000,
            gamma=0.99,
            pi_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            polyak=5e-3,
            adjust_entropy=False,
            alpha=0.2,
            target_entropy=-6.0,
            hidden1=256,
            hidden2=256,
            batch_size=64,
            pth=None,
            device='cpu',
            render='False'
            ):

    arg_dict = locals()

    num_ep = int(num_ep)
    buffer_size = int(buffer_size)

    env = gym.make(env_id)
    test_env = gym.make(env_id)

    dimS = env.observation_space.shape[0]   # dimension of state space
    nA = env.action_space.n                 # number of actions

    # (physical) length of the time horizon of each truncated episode
    # each episode run for t \in [0, T)
    # set for RL in semi-MDP setting

    agent = SACAgent(dimS,
                     nA,
                     env.action_map,
                     gamma,
                     pi_lr=pi_lr,
                     q_lr=q_lr,
                     alpha_lr=alpha_lr,
                     polyak=polyak,
                     adjust_entropy=adjust_entropy,
                     target_entropy=target_entropy,
                     alpha=alpha,
                     hidden1=hidden1,
                     hidden2=hidden2,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     device=device,
                     render=render
                     )

    # log setting
    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")

    if pth is None:
        # default location of directory for training log
        pth = './log/' + env_id + '/'

    os.makedirs(pth, exist_ok=True)
    current_time = time.strftime("%m_%d-%H%_M_%S")
    file_name = pth + 'prioritized_' + current_time
    log_file = open(file_name + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')
    eval_log_file = open(file_name + '_eval.csv',
                         'w',
                         encoding='utf-8',
                         newline='')

    logger = csv.writer(log_file)
    eval_logger = csv.writer(eval_log_file)

    # save parameter configuration
    with open('./log/' + env_id + '/SAC_' + current_time + '.txt', 'w') as f:
        for key, val in arg_dict.items():
            print(key, '=', val, file=f)

    OPERATION_HOUR = T * num_ep
    # 200 evaluations in total
    EVALUATION_INTERVAL = OPERATION_HOUR / 200
    evaluation_count = 0
    # start environment roll-out

    global_t = 0.
    counter = 0

    for i in range(num_ep):
        if global_t >= OPERATION_HOUR:
            break

        # initialize an episode
        s = env.reset()
        t = 0.  # physical elapsed time of the present episode
        info = None
        ep_reward = 0.
        while t < T:
            # evaluation is done periodically
            if evaluation_count * EVALUATION_INTERVAL <= global_t:
                result = agent.eval(test_env, T=14400, eval_num=3)
                log = [i] + result
                eval_logger.writerow(log)
                evaluation_count += 1

            if counter < fill_buffer:
                a = random.choice(env.action_map(s))
            else:
                a = agent.get_action(s)

            s_next, r, d, info = env.step(a)
            ep_reward += gamma ** t * r
            dt = info['dt']
            t = info['elapsed_time']
            global_t += dt
            counter += 1
            agent.buffer.append(s, a, r, s_next, False, dt)
            if counter >= start_train and counter % train_interval == 0:
                # training stage
                # single step per one transition observation
                for _ in range(train_interval):
                    agent.train()
            s = s_next

        # save training statistics
        log_time = datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
        op_log = env.operation_log
        print('+' + '=' * 78 + '+')
        print('+' + '-' * 31 + 'TRAIN-STATISTICS' + '-' * 31 + '+')
        print('{} (episode {}) reward = {:.4f}'.format(log_time, i, ep_reward))
        print('+' + '-' * 32 + 'FAB-STATISTICS' + '-' * 32 + '+')
        print('carried = {}/{}\n'.format(op_log['carried'], sum(op_log['total'])) +
              'remain_quantity : {}\n'.format(op_log['waiting_quantity']) +
              'visit_count : {}\n'.format(op_log['visit_count']) +
              'load_two : {}\n'.format(op_log['load_two']) +
              'unload_two : {}\n'.format(op_log['unload_two']) +
              'load_sequential : {}\n'.format(op_log['load_sequential']) +
              'total : ', op_log['total']
              )
        print('+' + '=' * 78 + '+')
        print('\n', end='')
        logger.writerow(
            [i, ep_reward, op_log['carried']]
            + op_log['waiting_quantity']
            + list(op_log['visit_count'])
            + [op_log['load_two'], op_log['unload_two'], op_log['load_sequential']]
            + list(op_log['total'])
        )
    log_file.close()
    eval_log_file.close()

    return


if __name__ == "__main__":
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True)
    parser.add_argument('--num_trial', required=False, default=1, type=int)
    parser.add_argument('--gamma', required=False, default=0.99, type=float)
    parser.add_argument('--tau', required=False, default=5e-3, type=float)
    parser.add_argument('--pi_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--q_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--alpha_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=1000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=1000, type=int)
    parser.add_argument('--batch_size', required=False, default=64, type=int)
    parser.add_argument('--buffer_size', required=False, default=1e6, type=float)
    parser.add_argument('--num_ep', required=False, default=1e3, type=float)
    parser.add_argument('--T', required=False, default=300, type=float)
    parser.add_argument('--alpha', required=False, default=0.2, type=float)
    parser.add_argument('--adjust_entropy', action='store_true')
    parser.add_argument('--target_entropy', required=False, default=-6.0, type=float)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--device', required=False, default=default_device)

    args = parser.parse_args()

    for _ in range(args.num_trial):
        run_sac(args.env,
                num_ep=args.num_ep,
                T=args.T,
                start_train=args.start_train,
                train_interval=args.train_interval,
                fill_buffer=args.fill_buffer,
                gamma=args.gamma,
                pi_lr=args.pi_lr,
                q_lr=args.q_lr,
                alpha_lr=args.alpha_lr,
                polyak=args.tau,
                adjust_entropy=args.adjust_entropy,
                target_entropy=args.target_entropy,
                alpha=args.alpha,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                device=args.device,
                render=args.render
                )
