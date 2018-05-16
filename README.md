# Imagination Augmented Agents

## Description
TensorFlow implementation of [Imagination Augmented Agents](https://arxiv.org/abs/1707.06203) by DeepMind, published as a conference proceeding at NIPS 2017. This paper combines model free and model
based RL in an agent that constructs implicit plans by learning to interpret
predictions from a learned environment model.

The agent creates its own internal simulations using a learned environment model. This environment model is trained
independently of the agent on future frame and reward prediction conditioned on
action using a baseline actor (a2c) to play out a number of
games. The environment model can then be used to simulate potential
trajectories for the imagination augmented agent. To efficiently use these
simulated trajectories, the agent learns an encoder that extracts information from these
imaginations including both state and reward.

![Network Architecture](https://github.com/ASzot/imagination-augmented-agents-tf/raw/master/img/arch.png "Network architecture")

Above is the network architecture diagram from the paper. Below is the
architecture of the environment model, also taken from the paper.

![Environment Model](https://github.com/ASzot/imagination-augmented-agents-tf/raw/master/img/env_model.png "Environment Model")

This implementation was done according to the specifications of the
architecture in the appendix of the paper for Pacman. The game used in this was
Pacman as used in the paper, however, any game could be substituted.

Code from OpenAI's Baseline is used for multiprocessing
(`common/multiprocessing_env.py`). DeepMind's implementation of MiniPacman
is used (`common/deepmind.py`). Implementation of `common/minipacman.py` is from @higgsfield.

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
