import os
import shutil
from glob import glob
from tqdm import tqdm


data_dirs = ['dataset1', 'dataset2', 'dataset3'] # folders with the collected data
res_dir = 'processed_data'
diff = 2 # speed difference for turns (Ka parameter in collect.py)
os.makedirs(res_dir, exist_ok=True)

# Speed-turn correspondence:
# 12 20 - left2
# 14 18 - left1 
# 16 16 - straight
# 18 14 - right1 
# 20 12 - right2 

for data_id, data_dir in enumerate(data_dirs):
    file_names = glob(os.path.join(data_dir, '*.jpg'))

    for f in tqdm(file_names):
        # Extract info from the file name.
        fname = f.split(os.sep)[-1]
        fname_split = fname.split('.jpg')[0].split('_')
        f_id = fname_split[0] # image id
        f_left = int(fname_split[1]) # left wheel speed
        f_right = int(fname_split[2]) # right weel speed

        # Detect turn according to wheels speed.
        if f_left == f_right:
            turn = 'straight'
        elif f_left-f_right == 2*diff:
            turn = 'right1'
        elif f_left-f_right == 4*diff:
            turn = 'right2'
        elif f_right-f_left == 2*diff:
            turn = 'left1'
        elif f_right-f_left == 4*diff:
            turn = 'left2'
        else:
            exit('Wrong file', fname)

        # Rename image and save to res_dir folder.
        new_id = '{:02d}'.format(data_id) + f_id[2:]
        new_fname = os.path.join(res_dir, '{}_{}.jpg'.format(new_id, turn))
        shutil.copy(f, new_fname)
