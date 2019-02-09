import os
import shutil
from glob import glob
from tqdm import tqdm


data_dir = 'processed_data'
res_dir = 'data_by_turn'
file_names = glob(os.path.join(data_dir, '*.jpg'))

# Create folders.
turns = ['left2', 'left1', 'straight', 'right1', 'right2']
for t in turns:
    dir_name = os.path.join(res_dir, t)
    os.makedirs(dir_name, exist_ok=True)

# Split the data.
for f in tqdm(file_names):
    fname = f.split(os.sep)[-1]
    fname_split = fname.split('.jpg')[0].split('_')
    f_id = fname_split[0] # image id
    f_turn = fname_split[1] # turn
    new_fname = os.path.join(res_dir, f_turn, '{}_{}.jpg'.format(f_id, f_turn))
    shutil.copy(f, new_fname)
