
import gymnasium
from typing import Dict, Any

def determine_gymnasium_API(env: gymnasium.Env)->int:
    env.reset()
    dummy_action = env.action_space.sample()  # Sample a random "dummy" action
    try:
        result = env.step(dummy_action)
    except (IndexError, ValueError):
        dummy_action = dummy_action.reshape(1, -1)
        result = env.step(dummy_action)
    env.reset()  # Reset the environment after the dummy step
    return len(result)

def remove_truncated_if_needed(object:Any, env: gymnasium.Env, io_map: Dict[str,str])->None:
    API = determine_gymnasium_API(env)
    if API == 4:
        object.truncated = False
        io_map.pop("truncated", None)
        return
    elif API == 5:
        return
    else:
        raise ValueError(f"Env API unknown step output is {API} long")