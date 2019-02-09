import os
import numpy as np
from glob import glob
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SteerDataSet(Dataset):
    """Track images dataset.
    root_folder - path to the images
    classes - list (ex: ['left2', 'left1', 'straight','right1', 'right2'])
    resize - dimension of resized image
    h_crop - crop everything above this coordinate
    transform - image transformation (ex: normalization)
    img_ext - image extension
    """
    def __init__(self, root_folder, classes, resize=84, h_crop=80,
                 transform=None, img_ext = ".jpg"):
        self.root_folder = root_folder
        self.resize = resize
        self.h_crop = h_crop
        self.transform = transform
        self.img_ext = img_ext        
        self.filenames = glob(os.path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        self.classes = classes
        self.codes = {} # digital codes for each of the classes (ex: left2 = 0)
        for idx, c in enumerate(classes):
            self.codes[c] = idx

    def __len__(self):        
        return len(self.filenames) # number of images in dataset
    
    def __getitem__(self,idx):
        # Load image, crop the top and resize.
        f = self.filenames[idx]        
        img = cv2.imread(f)
        img = img[self.h_crop:, :]
        img = cv2.resize(img, (self.resize, self.resize))
        SteerDataSet
        SteerDataSet
        SteerDataSet
        SteerDataSet
        SteerDataSet
        SteerDataSet
        SteerDataSet
        # Extract turn from file name and convert to digital code.
        turn = f.split(os.sep)[-1].split(self.img_ext)[0][7:]
        turn = self.codes[turn]
        turn = torch.tensor(int(turn), dtype=torch.long)
        
        return img, turn


def test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("dev_data/training_data",".jpg", transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for im, y in ds_dataloader:      
        print(im.shape)
        print(y)
        break


if __name__ == "__main__":
    test()