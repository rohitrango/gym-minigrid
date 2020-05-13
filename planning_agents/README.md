### Trajectory visualization app

This simple PyQt5 app aims to visualize the agent's trajectory.
Packages needed to install:
- PyQt5 (`pip install PyQt5`)
- Numpy
- Matplotlib
- gym_minigrid
- pickle



The data to be loaded should be a pickle file (*.pkl) in the following format:

``` python
data = [traj0, traj1 ... traj_n]
```

Each trajectory is of the following form:

```python
traj = {
	'obs': [obs_0, obs_1 .... obs_m],
	'act': [act_0, act_1 .... act_m],
	'rew': [rew_0, rew_1 .... rew_m]
}
```

All buttons are self-explanatory. Here are some descriptions of the collected trajectories:

- **fullobs** : contains the fully observable state for the ASIST agent to train on. The player has a limited field-of-view and ASIST has significantly more knowledge than the player in terms of state information.
- **agentobs**: contains the partial observable state for ASIST agent to train on. Here we assume that the field-of-view of ASIST is the same as that of the player (more realistic scenario). 
  - In both these directories, *preemptivev0.pkl* and *scouringv0.pkl* are the two strategies of players.
  - In both these directories, *me1.pkl* and *me2.pkl* are the two strategies corresponding to my behavior. 
- **agentobs40**: same as agentobs but with more irrelevant behavior added.



### Running the code

To run the code, simply install the required packages and type

```bash
python VisualizeApp.py
```

Click on **Choose File** to select any of the pkl files and use the buttons to navigate.



 