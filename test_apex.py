import ray
from ray.rllib.agents.dqn.apex import ApexTrainer,APEX_DEFAULT_CONFIG
import yaml

from ray import tune

with open("apex.yml",'r') as f:
    config=yaml.safe_load(f)["config"]

config.update({
    "framework":"torch",
    "evaluation_num_workers": 4,
    "evaluation_interval": 100,
    "evaluation_duration": 20,
    "evaluation_config": {
        # "num_gpus_per_worker": 0.01,
        "explore": False
    },
    "num_gpus": 1,
    "num_workers": 8,
    "env": "BreakoutNoFrameskip-v4",
})


stop = {
    "training_iteration": 100000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}


ray.init(num_cpus=32,num_gpus=1)
tune.run(ApexTrainer,config=config,stop=stop)