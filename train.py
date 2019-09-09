from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from agents import Agents
from ddpg import ddpg

resume = False
n_episodes=2000
eps_start=1.0
eps_end=0.01
eps_decay=0.995

'''
change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Banana.app"
Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
'''
env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agents = Agents(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=0)
scores = ddpg(env, brain_name, agents, n_episodes=n_episodes, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, resume=resume)

# plot the scores
plt.plot(np.arange(1, len(scores)+1), np.mean(scores, axis=-1))
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()   

# close the environment
env.close()
