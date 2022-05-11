#%%
from ruamel.yaml import YAML
from pathlib import Path
config_filepath=Path("config/atari-impala.yaml")
#%%
yaml=YAML(typ="safe")
custom_config=yaml.load(config_filepath)
custom_config
# %%

# %%
