#%%
import pickle
import os
from pathlib import Path
checkpoint_dir="~/ray_results/ImpalaTrainer_2022-04-27_20-43-00"


def load(checkpoint_dir):
    path=Path(checkpoint_dir).expanduser()
    path2=next(path.glob("ImpalaTrainer*"))
    path3=path2/'params.pkl'
    print(f"load from {path3}")
    with path3.open('rb') as f:
        params=pickle.load(f)
    return params

params=load(checkpoint_dir)

# %%
params["env"]
# %%
params2=load("~/ray_results/ImpalaTrainer_2022-05-14_00-48-19")
# %%
from ruamel.yaml import YAML
config_filepath="config/atari-impala.yaml"
iters_per_train=100

yaml=YAML(typ="safe")

custom_config=yaml.load(Path(config_filepath))

def display(params):
    for key in custom_config:
        if key in params:
            print(key,params[key])
        else:
            print(key, "not exist")
display(params)
print("="*20)
display(custom_config)
# %%
