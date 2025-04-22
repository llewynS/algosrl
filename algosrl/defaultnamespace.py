##################################################
# Core Constants (All Experiments Require These) #
##################################################
EPOCH = 'epoch'
STEP = 'step'
TRUNCATE = 'truncated'
INFO = 'info'
TERMINAL = 'terminal'
ACT = 'action'
ENVOBS = 'env_observation'
ENVOBSPRIME = 'env_observation_prime'
GOALREWARD = 'goal_reward'
ENVREWARD = 'env_reward'
RETURN = 'return'
GOAL = 'goal'

ENVVARS = [ENVOBS, ACT, ENVOBSPRIME]
DEFAULTEXPERIENCE = [ENVOBS, ACT, ENVOBSPRIME, ENVREWARD, TERMINAL, EPOCH , 'wait']