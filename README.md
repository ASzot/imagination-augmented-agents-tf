# Imagination Augmented Agents
Tensorflow implementation of <a href'https://arxiv.org/abs/1707.06203'>Imagination Augmented Agents</a> by DeepMind, published as a conference proceeding at NIPS 2017. 

## Description

## Prereqs
- TensorFlow 1.4.1
- NumPy 1.14.0

See `requirements.txt` for the complete list of requirements I used, however, only the two
listed above are important.

## Usage
Train the actor critic model using:
```
python a2c.py
```

Train the environment model using:
```
python env_model.py
```

Train the imagination augmented agent using:
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
