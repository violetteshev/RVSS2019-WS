import os
from glob import glob
import cv2
from tqdm import tqdm


opposites = {'left1': 'right1',
             'left2': 'right2',
             'right1': 'left1',
             'right2': 'left2'}

turn = 'left1' # specify what images to mirror
root_dir = 'data_by_turn'
new_turn = opposites[turn] # opposite turn
data_dir = os.path.join(root_dir, turn)
res_dir = os.path.join(root_dir, new_turn)
file_names = glob(os.path.join(data_dir, '*.jpg'))

for f in tqdm(file_names):
    fname = f.split(os.sep)[-1]
    fname_split = fname.split('.jpg')[0].split('_')
    f_id = fname_split[0]
    
    if f_id[2] == '1': # if image is already mirrored, don't use it
        continue
    
    img = cv2.imread(f)
    flipped_img = cv2.flip(img, 1) # flip the image
    new_id = f_id[0:2] + '1' + f_id[3:]
    new_fname = os.path.join(res_dir, '{}_{}.jpg'.format(new_id, new_turn))
    cv2.imwrite(new_fname, flipped_img)
