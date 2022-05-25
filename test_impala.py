# %%
from ref.impala import ImpalaTrainer, DEFAULT_CONFIG
from atari_wrappers import register_my_env
from pathlib import Path
from ray.rllib.utils import merge_dicts
from ray.rllib.models import ModelCatalog

from ray.rllib.evaluation import RolloutWorker

from ruamel.yaml import YAML


import ray
from ray import tune



# %%

config_filepath="config/atari-impala.yaml"
iters_per_train=100

yaml=YAML(typ="safe")

custom_config=yaml.load(Path(config_filepath))


config=merge_dicts(DEFAULT_CONFIG,custom_config)
config["timesteps_per_iteration"]=config["train_batch_size"]*iters_per_train

# =========== ENV =============

# config["env"]="BeamRiderNoFrameskip-v4"
# config["env"]="QbertNoFrameskip-v4"
config["env"]="SpaceInvadersNoFrameskip-v4"

# disable default rllib atari deepmind wrappers
config["preprocessor_pref"] = None
config["env"]=register_my_env(config["env"], max_episode_steps=None)


# ========= model ==============
from visonnet import VisionNetwork,VisionNetwork2
ModelCatalog.register_custom_model("MyVisonNetwork",VisionNetwork)
ModelCatalog.register_custom_model("MyVisonNetwork2",VisionNetwork2)


stop = {
    "training_iteration": 1000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}


# from ray.rllib.utils.debug import summarize

# print(summarize(config))
# exit()
cpus=config["num_workers"]+config["evaluation_num_workers"]+1

ray.init(
    include_dashboard=True,
    # local_mode=True,
    num_cpus=cpus,num_gpus=1)

tune.run(ImpalaTrainer,config=config,stop=stop,checkpoint_freq=10)
# results: ~/workspace/rllib-record/ImpalaTrainer_2022-04-17_19-08-00
# results2: /home/zaber/ray_results/ImpalaTrainer_2022-04-27_20-43-00

# %%
