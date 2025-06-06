# Cheat Sheet for Core Setup Functionality

### 1. Port Check

The first step is to figure out which USB port maps to each arm (follower/leader). First we need to enable access to the relevant ports:
```bash
sudo chmod 666 /dev/ttyACM0
```
then
```bash
sudo chmod 666 /dev/ttyACM1
```

Once the ports are opened, run this script and follow instructions:
```bash
python lerobot/scripts/find_motors_bus_port.py
```

Following this, update `lerobot/common/robot_devices/robots/configs.py` with the relevant ports for the leader and follower arm.

### 2. Camera Check
The next step is to ensure the cameras are connected correctly. Run this script to search for cameras connected:
```bash
python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
```

This should save some frames from each camera to the outputs directory. If it does not, you need to debug the camera setup.

### Calibration

The next step is to calibrate the robot arms. This only needs to be done once and the calibration is saved to the .cache directory. Should the calibration fail, retry and pay close attention to the arm poses at each position.
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

Images for the poses can be found at [here](lerobot/README.md)

### Teleop

To test the calibration has worked correctly, it's wise to teleoperate the robot and test the ranges of motion. This script allows teleop:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

### Record Dataset

Finally to record a dataset, use this command:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=outputs/test \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=5 \
  --control.push_to_hub=false
```
Make sure to change the command line arguments to match the desired configuration.

### Visualise a Dataset

The codebase provides a useful server to visualise a dataset locally:

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id outputs/*
``` 

### Train a Policy

1. Start a training run:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=outputs/* \
  --policy.type=act \
  --output_dir=outputs/train/* \
  --job_name=* \
  --policy.device=cuda \
  --wandb.enable=true
```
2. Resume a training run:
```bash
python lerobot/scripts/train_fcil_policy.py --config_path=/home/josh/phddev/lerobot/outputs/train/fcil_policy/success_only/checkpoints/last/pretrained_model/train_config.json --resume=true
```


### Run a Policy

```bash
python lerobot/scripts/control_robot.py   --robot.type=so100   --control.type=record   --control.fps=30   --control.single_task="Grasp a lego block and put it in the bin."   --control.repo_id=outputs/eval_success_100   --control.tags='["tutorial"]'   --control.warmup_time_s=5   --control.episode_time_s=30   --control.reset_time_s=30   --control.num_episodes=10   --control.push_to_hub=false   --control.policy.path=/home/josh/phddev/lerobot/outputs/train/fcil_policy/success_500/checkpoints/last/pretrained_model
```