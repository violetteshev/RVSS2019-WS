# RVSS2019-WS
Repo for the workshop part of the [Australian Centre for Robotic Vision Summer School RVSS2019](https://www.roboticvision.org/rvss2019)

## Data Collection
Use `on_robot/collect_data/collect.py` for collecting images. Robot control:
- Direction is defined by speed of left and right wheels: `ppi.set_velocity(left, right)`.
- General speed is set in `Kd` parameter, `Ka` defines how much speed is changed when robot is turning.
- Press **up** to go straight: `left = right = Kd`.
- Press **right** to turn right: `left += Ka; right -= Ka`.
- Press **left** to turn left: `left -= Ka; right += Ka`.
- When turning in the opposite direction, speed is first reset to `Kd`.
- Press **space** to stop: `ppi.set_velocity(0, 0)`.

Images are saved to `on_robot/collect_data/data`. Image naming example: 000001_18_14.jpg, where 000001 - image id, 18 - speed of the left wheel, 14 - speed of the right wheel.

## Data Preparation
1. Put all the collected data in folders `dataset1`, `dataset2`, etc (no more than 999 images per folder or there'll be problems with names).

2. Use `rename.py` to rename all images and put them to processed_data folder. The new naming: 010001_right2.jpg, where first two digits (01) identify dataset folder, las 4 digits (0001) - image id, string (right2) - type of turn.

3. Use `split_by_turn.py` to split the images to different folders according to the type of turn.

4. Use `data_augmentation.py` to mirror the images. If 3rd digit in image id is '1' then image was mirrored. (Ex: 030001_right1.jpg - original image; 031001_left1.jpg - mirrored image.)

5. Use `delete.py` to cut down the number of images (for class balance).

6. Use `create_dataset.py` to split images into training and validation sets, so that data for each turn is splitted 70/30.

7. Move `train_data` and `val_data` folders to `on_laptop/dev_data`.

## Training
- `on_laptop/steer_net/steerDS.py` contains `SteerDataSet` class for loading (image, label) dataset instances.
- `on_laptop/steer_net/steerNet.py` contains `steerNet` network with following architecture:
![](/info/network.png)
- `on_laptop/steer_net/train_model.py` contains function `train_model` that performs training and validation steps and saves the model that achieves the best accuracy on validation data.
- Run `on_laptop/train.py` to train the model. Output directory for model weights and information files is `on_laptop/trained_models/{random number}`. 

## Testing
1. Move model file `steerNet.py` and model weights `steerNet.pt` to `on_robot/deploy`.
2. Run `on_robot/deploy/deploy0.py`.