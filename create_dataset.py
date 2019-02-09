import os
import shutil
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split


data_dir = 'data_by_turn'
train_dir = 'train_data'
val_dir = 'val_data'
turns = ['left1', 'left2', 'right1', 'right2', 'straight']
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split images of each turn, such that 70% goes to training data, 30% - validation.
for turn in turns:
    file_names = glob(os.path.join(data_dir, turn, '*.jpg'))
    # Get indices for random train/val split.
    train, val = train_test_split(file_names, test_size=0.3)
    
    # Save training data.
    for f in tqdm(train):
        fname = f.split(os.sep)[-1]
        new_fname = os.path.join(train_dir, fname)
        shutil.copy(f, new_fname)
    
    # Save validation data.
    for f in tqdm(val):
        fname = f.split(os.sep)[-1]
        new_fname = os.path.join(val_dir, fname)
        shutil.copy(f, new_fname)
