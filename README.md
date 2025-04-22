# AlgOSRL: RL Module For AlgOS
This module utilises [AlgOS](https://github.com/llewynS/algos) as its core and provides an interface for implementing reinforcement learning algorithms. 

AlgOS provides standardised:
* experimental interface,
* self organising algorithmic segments via [AbstractParametered](https://github.com/llewynS/algos/blob/main/algos/interfaces/abstractbaseclasses/hyperparameter.py#L142) and [AbstractComponent](https://github.com/llewynS/algos/blob/main/algos/interfaces/abstractbaseclasses/component.py),
* logging (can replace tensorboard logging with internal logger as .record interface is same),
* central DB schema that supports PostgreSQL and MySQL (integrates with Optuna DB),
* factory methods for dependency injection,
* hyperparameter for automated hyperparameter optimisation, and
* input parsing for automated script generation. 

AlgOSRL extends this to work in the reinforcement learning space by:
* providing an RLExperiment and RLComponent interface to extend,
* a numpy replay buffer that offers indexed access,
* wrappers for goal based RL that work with the maze environments from [robotics_gymnasium](https://robotics.farama.org/),
* wrapper for [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) that allow the use of their off-policy agents, and
* an evaluator for goals based on their coverage. 

A wrapper for [Tianshou](https://github.com/thu-ml/tianshou) also provided but has limited testing due to its incompatibility with Dict based inputs with its replay buffer and the integration of its replay buffer. 

## Installation
Install via pip:
```sh
pip install algosrl
```

Install from the cloned directory:

```sh
pip install -e .
```

## Documentation
TBA

## Examples
The examples folder contains an example runner script (runs a single experiment) and an example optimisation script (runs a study to find optimal hyperparameters). The optimisation script will run locally by default. 

## Roadmap
* Provide Colab or Hugging face examples.
* Generate documentation.
* Include probabalistic curriculum learning code.


## Citing the Project

To cite this repository in publications:

```bibtex
@misc{algos,
      title={AlgOS: Algorithm Operating System}, 
      author={Llewyn Salt and Marcus Gallagher},
      year={2025},
      eprint={2504.04909},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2504.04909}, 
}
```