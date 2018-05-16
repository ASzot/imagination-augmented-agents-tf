# Imagination Augmented Agents

## Description
Tensorflow implementation of [Imagination AugmentedAgents]("https://arxiv.org/abs/1707.06203") by DeepMind, published as a conference proceeding at NIPS 2017. This paper combines model free and model
based RL in an agent that constructs implicit plans by learning to interpret
predictions from a learned environment model.

The agent has its own internal simulations using a learned environment model
(this is trained independently of the agent). This environment model is trained
independently of the agent on future frame and reward prediction conditioned on
action using a baseline actor (a2c) to simulate a number of
games. The environment model can then be used to simulate a number of
trajectories for the imagination augmented agent. To efficiently use these
simulations, the agent learns an encoder that extracts information from these
imaginations beyond simply rewards.

## Prerequisites
- TensorFlow 1.4.1
- NumPy 1.14.0

See `requirements.txt` for the complete list of requirements I used, however, only the two
listed above are important.

## Usage
Train the actor critic model using the following. Training this is a necessary
baseline and needs to be used to train the environment model.
```
python a2c.py
```

Train the environment model using the following. Remember the a2c model must be
trained already.
```
python env_model.py
```

Train the imagination augmented agent using the following. Remember the
environment model must already be trained. 
```
python i2a.py
```

Evaluate any agent (in the terminal). Change the model checkpoint to whichever
actor you would like to evaluate (either from i2a or a2c). 
```
python evaluate.py
```

To see the agent play the game visually run the `eval_actor_vis.ipynb` Jupyter
Notebook.

To see the imagined states from the environment model visually run the
`eval_env_model.ipynb` Jupyter Notebook.
