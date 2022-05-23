#%%
import pickle
import os
from pathlib import Path
checkpoint_dir="~/ray_results/ImpalaTrainer_2022-05-20_15-41-12/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit18000_38afd_00000_0_2022-05-20_15-41-12/"



path=Path(checkpoint_dir).expanduser()/'params.pkl'
with path.open('rb') as f:
    params=pickle.load(f)


# %%
params["env"]
# %%
