import pathlib
import argparse

from algos import AbstractExperiment, runexp, check_exps
from algosrl.concreteclasses.experiments import SACVanillaSB3RLExperiment, TD3VanillaSB3RLExperiment

if __name__ == '__main__':
    default_path = pathlib.Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='SACVanillaSB3RLExperiment')
    parser.add_argument('--path', default=default_path)
    parser.add_argument('--name', default='')
    args, unknowns = parser.parse_known_args()
    args = vars(args)
    strat = args.pop('strategy')
    if not check_exps(strat):
        raise NotImplementedError(f'{strat} is not implemented')
    exp = AbstractExperiment._register[strat]
    runexp(exp, args, unknowns)