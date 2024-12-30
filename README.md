 # Deep Reinforcement Learning Agents

 ## Setup

On Windows / Linux:

```bash
$ conda create -y -n drlnd python==3.6.13 pip
$ conda activate drlnd

$ cd python
$ pip install .

$ pip install pre-commit
$ pre-commit install
```

Create environment on Apple Silicon:

```bash
## create empty environment
conda create -y -n drlnd

## activate
conda activate drlnd

## use x86_64 architecture channel(s)
conda config --env --set subdir osx-64

## install python, numpy, etc. (add more packages here...)
conda install python==3.6.13 pip
```

- Place games in `./games/<game_name>`
- Create `.env` file and copy content from `.env_demo`. Adapt file names if necessary.

