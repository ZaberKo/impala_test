#%%
from ray.rllib.agents.trainer import with_common_config
from ruamel.yaml import YAML
from pathlib import Path
from ref.impala import ImpalaTrainer, DEFAULT_CONFIG
from ray.rllib.utils import merge_dicts
yaml=YAML(typ="safe")
config_filepath="config/atari-impala-old.yaml"
custom_config=yaml.load(Path(config_filepath))

config=merge_dicts(DEFAULT_CONFIG.copy(),custom_config)





# %%
from ray.rllib.models import ModelCatalog


from atari_wrappers import *

env_name="SpaceInvadersNoFrameskip-v4"

def env_creator(env_config):
    env = gym.make(env_name, **env_config)
    env = MyMonitorEnv(env)
    env = AtariPreprocessing(env,
                             noop_max=30,
                             frame_skip=4,
                             screen_size=84,
                             terminal_on_life_loss=False,
                             grayscale_obs=True,
                             grayscale_newaxis=True,
                             scale_obs=False,
                             )
    env = FrameStack(env, 4)
    return env

env=env_creator({})

#%%

model=ModelCatalog.get_model_v2(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=env.action_space.n,
    model_config=config["model"],
    framework="torch",
)


#%%
from visonnet import VisionNetwork,VisionNetwork2


mymodel_config=merge_dicts(DEFAULT_CONFIG.copy(),yaml.load(Path("config/atari-impala.yaml")))["model"]
print(mymodel_config)
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("MyVisonNetwork",VisionNetwork)
ModelCatalog.register_custom_model("MyVisonNetwork2",VisionNetwork2)
mymodel=ModelCatalog.get_model_v2(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=env.action_space.n,
    model_config=mymodel_config,
    framework="torch",
)

#%%
def calc_params(model):
    counts=0
    for params in model.parameters():
        counts+=params.numel()
    return counts

print(calc_params(model))
print(calc_params(mymodel))
#%%
import torch
obs=env.reset()
obs=torch.from_numpy(obs).unsqueeze(0)
print(obs.shape)
logits=model({"obs":obs})
value=model.value_function()
# %%

# %%
