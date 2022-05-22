#%%
import pickle
import os

checkpoint_dir="/home/zaber/ray_results/ImpalaTrainer_2022-05-20_15-41-12/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit18000_38afd_00000_0_2022-05-20_15-41-12"

path=os.path.join(checkpoint_dir,'params.pkl')
with open(path,'rb') as f:
    params=pickle.load(f)


# %%
