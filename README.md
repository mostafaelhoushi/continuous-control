[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"
[plot1]: ./plot.png "Trained PLot"

# Continuous Control using Deep Reinforcement Learning

### Introduction

This project uses [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.


![Trained Agent][image1]

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment contains 20 identical agents, each with its own copy of the environment.

The barrier for solving the environment is that the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 

### Getting Started

1. Clone the repo:
```
git clone https://github.com/mostafaelhoushi/continuous-control.git
```

2. Change directory into the repo:
```
cd continuous-control
```

3. Download the Unity environment using one of the commands below.  You need only select the environment that matches your operating system:
- Linux: 
```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
```
- Mac OSX:
```
curl -O https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
```
- Windows (32-bit): [PowerShell]
```
$client = new-object System.Net.WebClient
$client.DownloadFile("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip")
```
- Windows (64-bit):
```
$client = new-object System.Net.WebClient
$client.DownloadFile("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip")
```
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

4. Unzip (or decompress) the downloaded file: 
- Linux: 
```
unzip Reacher_Linux.zip
```
- Mac OSX:

```
unzip -a Reacher.app.zip
```
- Windows (32-bit): [PowerShell]
```
Expand-Archive -Path Reacher_Windows_x86.zip -DestinationPath .
```
- Windows (64-bit): [PowerShell]
```
Expand-Archive -Path Reacher_Windows_x86_64.zip -DestinationPath .
```

5. Create (and activate) a new environment with Python 3.6.
- Linux or Mac OSX:
```
conda create --name drlnd python=3.6
source activate drlnd
```
- Windows:
```
conda create --name drlnd python=3.6
activate drlnd
```

6. Install the OpenAI gym library and PyTorch libraries:
```
pip install gym torch torchvision
```

7. Install the 3rd party Python modules in the repo:
```
cd 3rdparty
pip install .
cd ..
```

8. Edit the `train.py` and `test.py` scripts by updating the `file_name` parameter in the first function call in the file. Read the comments in the file to determine what path you should use.

9. Run the training script:
```
python train.py
```

You should expect an output similar to this:
```
Found path: /home/melhoushi/Udacity/continuous-control/./Reacher_Linux/Reacher.x86_64
Mono path[0] = '/home/melhoushi/Udacity/continuous-control/./Reacher_Linux/Reacher_Data/Managed'
Mono config path = '/home/melhoushi/Udacity/continuous-control/./Reacher_Linux/Reacher_Data/MonoBleedingEdge/etc'
Preloaded 'ScreenSelector.so'
Preloaded 'libgrpc_csharp_ext.x64.so'
Unable to preload the following plugins:
	ScreenSelector.so
	libgrpc_csharp_ext.x86.so
Logging to /home/melhoushi/.config/unity3d/Unity Technologies/Unity Environment/Player.log
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
Number of agents: 20
Size of each action: 4
There are 20 agents. Each observes a state with length: 33
The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
Episode 100	Average Score: 28.92age Score: 28.92
Episode 104	Score: 36.18	Average Score: 30.34
Environment solved in 4 episodes!	Average Score: 30.34
```

and finally you should find the following plot:
![Training Plot][plot1]

10. Click on the "X" button to close the plotting window.

11. You should now find a `solution.pth` file in your directory that contains the PyTorch models of your training agent.

12. You can now run the testing script:
```
python test.py
```


### Notebook
You may also explore and run the `Continuous_Control.ipynb` notebook.

1. Install the 3rd party Python modules in the repo:
```
cd 3rdparty
pip install .
cd ..
```

2. Create an IPython kernel for the `drlnd` environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

3. Start Jupyter Notebook:
```
jupyter notebook
```

4. Copy the URL message that you see and paste it into your browser.

5. In the browser, open the `Continuous_Control.ipynb` notebook.

6. Before running code in a notebook, change the kernel to match the drlnd environment by clicking on the top menu: Kernel -> Change Kernel -> drlnd
