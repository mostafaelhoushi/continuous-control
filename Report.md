# Report
[plot1]: ./plot.png "Trained PLot"

## Learning Algorithm
To train the agents we used Deep Deterministic Policy Gradients (DDPG) algorithm.
At its core, DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. 
DDPG is an actor-critic algorithm as well; it primarily uses two neural networks, one for the actor and one for the critic

### Chosen Hyperparameters

- Learning Rate: 1e-4 (in both DNN)
- Batch Size: 128
- Replay Buffer: 1e5
- Gamma: 0.99
- Tau: 1e-3
- Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)

### Model Architecture
A typical fully-connected neural network architecture consists of cascaded pairs of linear and non-linear layers. 

For both the Actor and Critic, the size of the input linear layer is the state size, and the size of the output linear layer is the number of possible actions. The output of the Actor is concatenated to the Critic's first layer output to be connected to the first hidden layer's input.

Our `Actor` and `Critic` classes defined in `model.py` allow for an arbitrary number of layers with arbitrary sizes to be defined by the user. 

For the example scripts and checkpoint file provided, we used those layers structures:
- Actor
	- Hidden: (input, 256) - ReLU
	- Hidden: (256, 128) - ReLU
	- Output: (128, 4) - TanH
- Critic
	- Hidden: (input, 256) - ReLU
	- Hidden: (256 + action_size, 128) - ReLU
	- Output: (128, 1) - Linear

## Plot of Rewards
The following is the plot of reward at every episode:
![Training Plot][plot1]

## Ideas for Future Work

- Increase number of training episodes
- Increase depth of each network
