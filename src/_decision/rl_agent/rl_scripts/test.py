import zipfile
import pickle

ckpt_path = "/home/chendawww/workspace/rl-navibot/tries/burger_navi_in_world/saved_models/SAC/sac_nav_model_136000_steps.zip"
with zipfile.ZipFile(ckpt_path, 'r') as z:
    with z.open('data') as f:
        content = f.read().decode('utf-8')
        print(content)

# import zipfile

# ckpt_path = "/home/chendawww/workspace/rl-navibot/tries/burger_navi_in_world/saved_models/SAC/sac_nav_model_136000_steps.zip"
# with zipfile.ZipFile(ckpt_path, 'r') as z:
#     for name in z.namelist():
#         print(name)
