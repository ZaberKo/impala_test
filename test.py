#%%
import ray.cloudpickle as pickle
from pathlib import Path
checkpoint=Path("~/ray_results/ImpalaTrainer_2022-05-08_01-25-34/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit40000_b432f_00000_0_2022-05-08_01-25-35/checkpoint_000800/checkpoint-800").expanduser()
checkpoint_meta=Path("~/ray_results/ImpalaTrainer_2022-05-08_01-25-34/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit40000_b432f_00000_0_2022-05-08_01-25-35/checkpoint_000800/checkpoint-800.tune_metadata").expanduser()

#%%
with checkpoint.open("rb") as f:
    data=pickle.load(f)

with checkpoint_meta.open('rb') as f:
    metadata=pickle.load(f)
# %%
list(data.keys())
# %%
worker_state=pickle.loads(data["worker"])
list(worker_state.keys())
# %%
print(worker_state["filters"])
print(worker_state["policy_specs"].values())
# %%
for name,data in worker_state["policy_specs"].items():
    print(name)
    print(data)
# %%

# %%
