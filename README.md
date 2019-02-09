# RVSS2019-WS
Repo for the workshop part of the [Australian Centre for Robotic Vision Summer School RVSS2019](https://www.roboticvision.org/rvss2019)

## Data Preparation
1. Put all the collected data in folders dataset1, dataset2, etc (no more than 999 images per folder or there'll be problems with names). Image naming example: 000001_18_14.jpg, where 000001 - image id, 18 - speed of the left wheel, 14 - speed of the right wheel.

2. Use rename.py to rename all images and put them to processed_data folder. The new naming: 010001_right2.jpg, where first two digits (01) identify dataset folder, las 4 digits (0001) - image id, string (right2) - type of turn.

3. Use split_by_turn.py to split the images to different folders according to the type of turn.

4. Use data_augmentation.py to mirror the images. If 3rd digit in image id is '1' then image was mirrored. (Ex: 030001_right1.jpg - original image; 031001_left1.jpg - mirrored image.)

5. Use delete.py to cut down the number of images (for class balance).

6. Use create_dataset.py to split images into training and validation sets, so that data for each turn is splitted 70/30.

7. Move train_data and val_data folders to on_laptop/dev_data.