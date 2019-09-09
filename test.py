import numpy as np
import torch
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from agents import Agents

'''
change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Banana.app"
Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
'''
env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86")

brain_name = env.brain_names[0]                    # get the default brain
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
num_agents = len(env_info.agents)                  # get number of agents
states = env_info.vector_observations              # get the current state
state_size = states.shape[1]
action_size = brain.vector_action_space_size
agents = Agents(num_agents=num_agents,             # create RL agents
                state_size=state_size, 
                action_size=action_size, random_seed=0) 

checkpoint = torch.load('solution.pth', map_location={'cuda:0': 'cpu'})          # load weights
agents.actor_local.load_state_dict(checkpoint['actor_local_state_dict'])
agents.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
agents.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
agents.critic_local.load_state_dict(checkpoint['critic_local_state_dict'])
agents.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
agents.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

scores = np.zeros(num_agents)                          # initialize the score (for each agent)
eps = 0.1
while True:
    actions = agents.act(states, eps)                  # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# close the environment
env.close()
