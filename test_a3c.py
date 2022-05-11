import ray
from ray import tune

# %%
from ray.rllib.agents.a3c.a2c import A2CTrainer, A2C_DEFAULT_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts
import torch
import ray.rllib.env.wrappers.atari_wrappers 

def actor_critic_loss(policy, model, dist_class, train_batch, reduce_op=torch.sum):
    logits, _ = model(train_batch)
    values = model.value_function()

    valid_mask = torch.ones_like(values, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    pi_err = -reduce_op(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES],
                            valid_mask))

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * reduce_op(
            torch.pow(
                torch.masked_select(
                    values.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0

    entropy = reduce_op(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.entropy_coeff)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss


class _A2CTrainer(A2CTrainer):
    def get_default_policy_class(self, config):
        return A3CTorchPolicy.with_updates(loss_fn=actor_critic_loss)


config = merge_dicts(A2C_DEFAULT_CONFIG, {
    "framework": "torch",
    "num_gpus": 0.1,
    "lambda": 0.95,
    "num_workers": 5,
    "num_envs_per_worker": 5,
    "rollout_fragment_length": 20,
    "train_batch_size": 200,
    "min_time_s_per_reporting": 0,
    "evaluation_num_workers": 4,
    "evaluation_interval": 10,
    "evaluation_duration": 100,
    # "evaluation_config":{
    #     "num_gpus_per_worker": 0.01,
    #     "explore":False
    # },
    "clip_rewards": True,
    "lr_schedule": [
        [0, 0.0007],
        [20000000, 0.000000000001]
    ],
    "entropy_coeff": 0.1,
    # "env":"ALE/Breakout-v5"
    # "env":"CartPole-v1"
    "env": "BreakoutNoFrameskip-v4"
})


stop = {
    "training_iteration": 100000,
    # "timesteps_total": 1000000,
    # "episode_reward_mean": args.stop_reward,
}

ray.init(num_cpus=4,
local_mode=True
)
# results = tune.run(_A2CTrainer, config=config, stop=stop)
trainer = _A2CTrainer(config=config)
for i in range(5):
    trainer.train()
    print(f"iter {i} complete")
# %%
# result: /home/zaber/ray_results/_A2CTrainer_2022-03-24_12-25-18
