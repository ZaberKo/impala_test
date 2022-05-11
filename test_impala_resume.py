# %%
from ray.rllib.agents.impala import ImpalaTrainer, DEFAULT_CONFIG

from glom import glom
import torch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy

from ray.tune.trial import Trial
import ray
from ray import tune
ray.init(
    # include_dashboard=True,
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
    "evaluation_interval": 1e10,
    "evaluation_duration": 100,
    "evaluation_config": {
        # "num_gpus_per_worker": 0.01,
        # "num_envs_per_worker": 1,
        "explore": False
    },
    "env": "BreakoutNoFrameskip-v4",
    "rollout_fragment_length": 50,
    "train_batch_size": 500,

    "lr_schedule": [
        [0, 0.0005],
        [20000000, 0.000000000001]],
    # "min_time_s_per_reporting":0,
    "timesteps_per_iteration":500*100,
    # "log_level": "DEBUG"
})


stop = {
    "training_iteration": 250,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}

tune.run(ImpalaTrainer,config=config,stop=stop,checkpoint_freq=1,
# keep_checkpoints_num=1,
# resume=True
# restore="~/ray_results/ImpalaTrainer_2022-04-17_19-08-00/ImpalaTrainer_BreakoutNoFrameskip-v4_a4be1_00000_0_2022-04-17_19-08-00/checkpoint_000086/checkpoint-86"
restore="~/ray_results/ImpalaTrainer_2022-04-19_12-12-17/ImpalaTrainer_BreakoutNoFrameskip-v4_e6b7f_00000_0_2022-04-19_12-12-18/checkpoint_000213/checkpoint-213"
    )
# results: ~/workspace/rllib-record/ImpalaTrainer_2022-04-17_19-08-00


# %%
# trainer = ImpalaTrainer(config=config)
# policy=trainer.get_policy()
# print(policy.model)
# print(policy.is_recurrent())


def summary(res):
    evaluation=glom(res,"evaluation",default=None)
    episode_len_mean=glom(evaluation,"episode_len_mean",default=None)
    episode_reward_mean=glom(evaluation,"episode_reward_mean",default=None)
    episode_hist=glom(evaluation,"hist_stats",default=None)
    iteration = glom(res, "training_iteration")
    timesteps_total = glom(res, "timesteps_total")
    timesteps_this_iter = glom(res, "timesteps_this_iter")

    learner_info=glom(res,"info.learner.default_policy.learner_stats")
    policy_entropy=glom(learner_info,"entropy")
    policy_loss=glom(learner_info,"policy_loss")
    vf_loss=glom(learner_info,"vf_loss")
    return {
        "timesteps_total": timesteps_total,
        "timesteps_this_iter": timesteps_this_iter,
        "policy_entropy":policy_entropy,
        "policy_loss":policy_loss,
        "vf_loss":vf_loss,
        "episode_len_mean": episode_len_mean,
        "episode_reward_mean": episode_reward_mean,
        "episode_hist": episode_hist,
        "iteration": iteration,
    }


# %%
# for i in range(100000):
#     res = trainer.train()
#     res=summary(res)
#     if res["episode_len_mean"]:
#         pprint(res)
#         print("="*20)
