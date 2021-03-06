# %%
from impala_student2 import ImpalaStudentTrainer, STUDENT_DEFAULT_CONFIG
from atari_wrappers import register_my_env
from pathlib import Path

from ruamel.yaml import YAML


import ray
from ray import tune
ray.init(
    # include_dashboard=True,
    num_cpus=33,num_gpus=1)
# %%

config_filepath="config/atari-impala-teacher2.yaml"
iters_per_train=100

config = STUDENT_DEFAULT_CONFIG.copy()

yaml=YAML(typ="safe")

custom_config=yaml.load(Path(config_filepath))


config.update(custom_config)
config["timesteps_per_iteration"]=config["train_batch_size"]*iters_per_train


config["preprocessor_pref"] = None
config["env"]=register_my_env(config["env"], max_episode_steps=None)


stop = {
    "training_iteration": 1000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}


tune.run(ImpalaStudentTrainer,config=config,stop=stop,checkpoint_freq=10)

# %%
# trainer = ImpalaStudentTrainer(config=config)

# trainer.train()