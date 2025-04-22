import argparse
import json
from algos.scriptbuilders.optimiser import runopt
import pathlib
from algosrl.concreteclasses.experiments import SACVanillaSB3RLExperiment, TD3VanillaSB3RLExperiment, SACUniformExperiment, TD3UniformExperiment

if __name__ == '__main__':
    default_path = pathlib.Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()
    default_storage = f'sqlite:///{default_path}/primary.db'
    # default_storage = None
    parser.add_argument('--strategy', default='SACUniformExperiment')
    parser.add_argument('--storage', default=default_storage)
    parser.add_argument('--path', default=default_path)
    parser.add_argument('--name', default='')
    parser.add_argument('--computer', default='local')
    parser.add_argument('--ssh-name', default=None)
    parser.add_argument('--remote-path', default=None)
    parser.add_argument('--remote-prompt', default=None)
    parser.add_argument('--sbatch-config', default=None)
    parser.add_argument('--config-kwargs', default=None, type=str)
    parser.add_argument('--study-name', default=None)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--num-jobs', type=int, default=4)
    parser.add_argument('--sampler', default='tpe', choices=['tpe', 'gp', 'cmaes'])
    parser.add_argument('--runner-script', default=default_path / 'myexprunner.py')
    args, unknowns = parser.parse_known_args()
    args = vars(args)
    remote_prompt = args.pop('remote_prompt', None)
    if remote_prompt is not None:
       remote_prompt = r'' + remote_prompt.replace(r'\\\\', r"\\")
    remote_path = args.pop('remote_path', None)
    sbatch_config = args.pop('sbatch_config', None)
    config_kwargs = args.pop('config_kwargs', None)
    if config_kwargs is not None:
        try:
            config_kwargs = json.loads(config_kwargs)
        except json.JSONDecodeError:
            raise ValueError("config_kwargs should be a valid JSON string")
    ssh_name = args.pop('ssh_name', None)
    if args["computer"]=="remote":
        args.pop('runner_script', None)

    runopt(args, unknowns, remote_prompt, remote_path, sbatch_config, config_kwargs, ssh_name)
