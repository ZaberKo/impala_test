# %%
from ref.impala import ImpalaTrainer, DEFAULT_CONFIG
from atari_wrappers import register_my_env
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


config.update(custom_config)
config["timesteps_per_iteration"]=config["train_batch_size"]*iters_per_train


config["env"]="BeamRiderNoFrameskip-v4"
config["env"]=register_my_env(config["env"],max_episode_steps=3600*5)


stop = {
    "training_iteration": 1000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}

# %%
# or training mannually
trainer = ImpalaTrainer(config=config)
for i in range(stop["training_iteration"]):
    res = trainer.train()
    # del res["config"]
    # del res["hist_stats"]
    # print(summarize(res))
    print(f"current iter={i} sampled_ts={res['timesteps_total']} trained_ts={res['timesteps_since_restore']}")

