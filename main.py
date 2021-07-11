import gym
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch
from gamepad_agent import GamePadAgent
import seaborn as sns
import csv

ENV_NAME = "FetchPush-v1"
Train = False
Play_FLAG = False
With_DEMOES = True
# -------
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 40
num_updates_demonstration = 80
MAX_EPISODES = 2
DEMONSTRATION_EPISODES = 50
# -------
memory_size = 7e+5 // 50
batch_size = 256
demonstrator_lr = 1e-2  # -------
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'


def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
        env_dictionary = env_.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.linalg.norm(ag - g) <= 0.05:
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0
        for t in range(50):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, _, info_ = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


# ---------------------------------------------------------------------------------
def train_with_demos(agent, env):
    agent_gamepad = GamePadAgent()
    memory_buffer = []
    total_success_rate = []
    episode_success_rate = []
    for episode in range(DEMONSTRATION_EPISODES):
        per_success_rate = []
        episode_dict = {
            "state": [], "action": [], "info": [], "achieved_goal": [], "desired_goal": [], "next_state": [],
            "next_achieved_goal": []
        }
        env_dict = env.reset()
        state = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]
        ep_r = 0
        for t in range(50):
            if MPI.COMM_WORLD.Get_rank() == 0:
                action = agent_gamepad.action()
                # action = env.action_space.sample()
            else:
                action = agent.choose_action(state, desired_goal)
            next_env_dict, reward, done, info = env.step(action)

            next_state = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict["desired_goal"]
            per_success_rate.append(info['is_success'])
            ep_r += reward

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
            if MPI.COMM_WORLD.Get_rank() == 0:
                env.render()
                time.sleep(0.1)

        episode_dict["state"].append(state.copy())
        episode_dict["achieved_goal"].append(achieved_goal.copy())
        episode_dict["desired_goal"].append(desired_goal.copy())
        episode_dict["next_state"] = episode_dict["state"][1:]
        episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
        memory_buffer.append(dc(episode_dict))
        if MPI.COMM_WORLD.Get_rank() == 0:
            episode_success_rate.append(per_success_rate)
            local_success_rate = np.mean(episode_success_rate[-1])
            total_success_rate.append(local_success_rate)
            plot_demonstration_quality(total_success_rate)

    agent.store(memory_buffer)
    for n_update in range(num_updates_demonstration):
        agent.train_demonstrator()

    agent.update_networks_demonstrator()
    train_model(agent, env)


def train_model(agent, env):
    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                for t in range(50):
                    action_policy = agent.choose_action(state, desired_goal)
                    action_demonstrator = agent.choose_action_demonstrator(state, desired_goal)
                    action = agent.evaluate_action(state, desired_goal, action_policy, action_demonstrator)
                    next_env_dict, reward, done, info = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]

                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            agent.store(mb)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss
            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss / num_updates
            agent.update_networks()

        ram = psutil.virtual_memory()
        success_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        if MPI.COMM_WORLD.Get_rank() == 0:
            t_success_rate.append(success_rate)
            print(f"Epoch:{epoch}| "
                  f"Running_reward:{running_reward[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Success rate:{success_rate:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            agent.save_weights()

    if MPI.COMM_WORLD.Get_rank() == 0:
        plot_results(t_success_rate)


def plot_demonstration_quality(scores, x=None, ):
    sns.set()
    N = len(scores)
    running_avg = np.array(scores)
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Success Rate')
    plt.xlabel('Episodes')
    plt.plot(x, running_avg)
    filename = 'Demonstrations.png'
    plt.savefig(filename)


def plot_results(t_success_rate):
    plt.figure()
    sns.set()

    x = []
    y = []
    with open('FetchPickAndPlace_baseline.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))
    y = np.array(y)
    x = np.array(x)
    plt.plot(x, y, label='DDPG+HER', color='blue')

    plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate, label='Ours', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Success Rate')
    plt.title(ENV_NAME)
    plt.legend()
    plt.savefig("success_rate.png")
    plt.show()


def play_model(agent, env):
    player = Play(env, agent, max_episode=100)
    player.evaluate()


def main():
    env = gym.make(ENV_NAME)
    env.seed(MPI.COMM_WORLD.Get_rank())
    random.seed(MPI.COMM_WORLD.Get_rank())
    np.random.seed(MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(MPI.COMM_WORLD.Get_rank())
    agent = Agent(n_states=state_shape,
                  n_actions=n_actions,
                  n_goals=n_goals,
                  action_bounds=action_bounds,
                  capacity=memory_size,
                  action_size=n_actions,
                  batch_size=batch_size,
                  demonstrator_lr=demonstrator_lr,
                  actor_lr=actor_lr,
                  critic_lr=critic_lr,
                  gamma=gamma,
                  tau=tau,
                  k_future=k_future,
                  env=dc(env))

    if Train:
        train_model(agent, env)
    elif With_DEMOES:
        train_with_demos(agent, env)
    elif Play_FLAG:
        play_model(agent, env)


if __name__ == '__main__':
    main()
