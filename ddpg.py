import shutil
from collections import deque
import numpy as np
import torch

from agents import Agents

def ddpg(env, brain_name, agents, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, resume=False):
    scores_list = []                   # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    start_episode = 1

    if resume is True:
        checkpoint = torch.load('checkpoint.pth')
        start_episode = checkpoint['i_episode']
        scores_list = checkpoint['scores_list']
        scores_window = checkpoint['scores_window']
        eps = max(eps_end, eps_start**(i_episode-1-1))
        
        agents.actor_local.load_state_dict(checkpoint['actor_local_state_dict'])
        agents.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agents.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agents.critic_local.load_state_dict(checkpoint['critic_local_state_dict'])
        agents.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agents.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
    for i_episode in range(start_episode, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(agents.num_agents)                   # initialize the score (for each agent)
        while True:
            actions = agents.act(states, eps)                  # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)  
            dones = env_info.local_done                        # see if episode finished
            agents.step(states, actions, rewards, next_states, dones) # agents store or learn from the expreiences
            states = next_states                               # roll over states to next time step
            scores += rewards                             # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break 
        scores_window.append(scores)        # save most recent scores (for each agent)
        scores_list.append(scores)               # save most recent scores (for each agent)
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon
        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores), np.mean(scores_window)), end="")
        torch.save({
                'i_episode': i_episode + 1,
                'actor_local_state_dict': agents.actor_local.state_dict(),
                'actor_target_state_dict': agents.actor_target.state_dict(),
                'actor_optimizer_state_dict': agents.actor_optimizer.state_dict(),
                'critic_local_state_dict': agents.critic_local.state_dict(),
                'critic_target_state_dict': agents.critic_target.state_dict(),
                'critic_optimizer_state_dict': agents.critic_optimizer.state_dict(),
                'scores_list': scores_list,
                'scores_window': scores_window,
                }, 'checkpoint.pth')

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            shutil.copyfile('checkpoint.pth', 'solution.pth')
            break

    return scores_list
