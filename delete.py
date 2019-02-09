import os
import shutil
from glob import glob
from tqdm import tqdm


root_dir = 'data_by_turn'
turn = 'straight' # specify what images to cut down
data_dir = os.path.join(root_dir, turn)
res_dir = os.path.join(root_dir, turn+'-extra')
os.makedirs(res_dir, exist_ok=True)
file_names = glob(os.path.join(data_dir, '*.jpg'))

idx = 0
# Move every second image to 'extra' folder.
for f in tqdm(file_names):
    idx += 1
    if idx % 2 == 0:
        continue
    fname = f.split(os.sep)[-1]
    new_fname = os.path.join(res_dir, fname)
    shutil.move(f, new_fname)
