# %%
from ref.impala import ImpalaTrainer, DEFAULT_CONFIG
from atari_wrappers import register_timelimit_env
from pathlib import Path

from ruamel.yaml import YAML


import ray
from ray import tune
ray.init(
    # include_dashboard=True,
    num_cpus=33,num_gpus=1)
# %%

config_filepath="config/atari-impala.yaml"
iters_per_train=100

config = DEFAULT_CONFIG.copy()

yaml=YAML(typ="safe")

custom_config=yaml.load(Path(config_filepath))

fix_lr_config=yaml.load(Path("config/fix_lr.yaml"))
custom_config.update(fix_lr_config)


config.update(custom_config)
config["timesteps_per_iteration"]=config["train_batch_size"]*iters_per_train


config["env"]="PongNoFrameskip-v4"
config["env"]=register_timelimit_env(config["env"],max_episode_steps=3600*5)




stop = {
    "training_iteration": 1000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}


tune.run(ImpalaTrainer,config=config,stop=stop,checkpoint_freq=10)
# results: ~/workspace/rllib-record/ImpalaTrainer_2022-04-17_19-08-00
# results2: /home/zaber/ray_results/ImpalaTrainer_2022-04-27_20-43-00

