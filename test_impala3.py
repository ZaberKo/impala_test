# %%
from ref.impala2 import ImpalaTrainer, DEFAULT_CONFIG
import ray
from ray import tune
ray.init(
    include_dashboard=True,
    num_cpus=32,num_gpus=1)
# %%

config = DEFAULT_CONFIG.copy()
config.update({
    "framework": "torch",
    "num_gpus": 0.1,
    "num_workers": 16,
    "num_envs_per_worker": 5,
    "clip_rewards": True,
    "evaluation_num_workers": 4,
    "evaluation_interval": 1,
    "evaluation_duration": 100,
    "evaluation_config": {
        # "num_gpus_per_worker": 0.01,
        "explore": False
    },
    "env": "BreakoutNoFrameskip-v4",
    "rollout_fragment_length": 50,
    "train_batch_size": 500,
    "num_multi_gpu_tower_stacks": 1,

    "lr_schedule": [
        [0, 0.0005],
        [20000000, 0.000000000001]],
    "min_time_s_per_reporting":10,
    "timesteps_per_iteration":0,
    "log_level": "WARN"
})


#%%
stop = {
    "training_iteration": 100000,
}

tune.run(ImpalaTrainer,config=config,stop=stop)


# %%
# or training mannually
# 

